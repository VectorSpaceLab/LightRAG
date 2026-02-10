import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.file_utils import ModelOutput
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from dataclasses import dataclass
from accelerate import Accelerator

PH_TOKEN_ID = 100
INPUT_TAG = "[INPUT_RmehNsY1]"
CONTEXT_TAG = "[CONTEXT_RmehNsY1]"

@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    gold: Optional[Tensor] = None
    query_input_ids: Optional[Tensor] = None
    passage_input_ids: Optional[Tensor] = None
    length: Optional[int] = None

class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        model_name: str = None,
        peft_model_name: str = None,
        normalized: bool = True,
        pooling_method: str = 'cls',
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        lora_tune: bool = False,
        enable_token_level_retrieval: bool = False, # training arguments
        attn_implementation: str = "flash_attention_2",
        accelerator: Optional[Accelerator] = None,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation=attn_implementation)
        if peft_model_name:
            self.model = PeftModel.from_pretrained(self.model, peft_model_name, dtype=torch.bfloat16, is_trainable=False)
            self.model = self.model.merge_and_unload()
        
        if accelerator:
            self.accelerate = accelerator
            self.model.to(accelerator.device)
        
        if lora_tune:
            self.lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "gate_proj", "up_proj"],
                lora_dropout=0.03,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                # modules_to_save=["embed_tokens", "norm", "input_layernorm", "post_attention_layernorm"],
            )
            print('Loading LoRA Config...')
            self.model = get_peft_model(self.model, self.lora_config)
            self.trainable_params = "embed,norm"
            # enable trainable params
            [p.requires_grad_() for n, p in self.model.named_parameters() if any([k in n for k in self.trainable_params.split(",")])]
            self.model.print_trainable_parameters()

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.normalized = normalized
        self.pooling_method = pooling_method
        self.temperature = temperature
        if not normalized:
            self.temperature = 1.0

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.enable_token_level_retrieval = enable_token_level_retrieval

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.config.use_cache = False
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()
    
    def embedding(self, hidden_state, mask):
        if self.pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.pooling_method == 'cls':
            return hidden_state[:, -1]

    def encode(self, features, is_passage=False):
        if features["input_ids"].dim() == 1:
            features = {
                "input_ids": features["input_ids"].unsqueeze(0),
                "attention_mask": features["attention_mask"].unsqueeze(0),
            }
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        if self.normalized:
            last_hidden_state = torch.nn.functional.normalize(psg_out.last_hidden_state, dim=-1)
        else:
            last_hidden_state = psg_out.last_hidden_state
        if is_passage:
            return last_hidden_state
        
        p_reps = self.embedding(last_hidden_state, features['attention_mask'])
        return p_reps.contiguous()

    def singlemaxloss(self, scores, tagging):
        scores = F.log_softmax(scores, dim=1)
        scores = scores * tagging
        min_score, _ = torch.min(scores, dim=1)
        loss = -min_score[0]
        return loss

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))
    
    def multi_positive_nll_loss(self, scores, tags, total_pos_num):
        scores = F.log_softmax(scores, dim=1)
        loss = -torch.sum(scores * tags) / total_pos_num
        return loss

    def forward(self, query, passage, tags=None, gold_indices=None):
        if self.training:
            if self.enable_token_level_retrieval:
                q_reps = self.encode(query)
                p_reps = self.encode(passage, is_passage=True)

                token_emb = p_reps.squeeze(dim=0)
                scores = self.compute_similarity(q_reps, token_emb)
                scores = scores / self.temperature
                total_pos_num = gold_indices.size(1)

                loss = self.multi_positive_nll_loss(scores, tags, total_pos_num)
                return EncoderOutput(
                    loss=loss
                )
            else:
                q_reps = self.encode(query) # (batch_size, dim)
                p_reps = self.encode(passage) # (batch_size * num, dim)
                
                if self.negatives_cross_device:
                    q_reps = self._dist_gather_tensor(q_reps)
                    p_reps = self._dist_gather_tensor(p_reps)

                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores / self.temperature
                scores = scores.view(q_reps.size(0), -1)

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * (p_reps.size(0) // q_reps.size(0))
                loss = self.cross_entropy(scores, target)
                return EncoderOutput(
                    loss=loss,
                    scores=scores,
                    q_reps=q_reps,
                    p_reps=p_reps,
                )
        else:
            with torch.no_grad():
                self.model.eval()        
                q_reps = self.encode(query) 
                batch_size = passage["input_ids"].size(0)
                scores_batch = []

                # process in a loop due to OOM
                for i in range(batch_size):
                    single_passage = {
                        k: v[i:i+1] for k, v in passage.items() 
                        if isinstance(v, torch.Tensor)
                    }
            
                    p_rep_single = self.encode(single_passage, is_passage=True)
            
                    if q_reps.dim() == 3:    # [batch_size, 1, hidden_dim]
                        q_emb = q_reps[i]
                    elif q_reps.dim() == 2:  # [batch_size, hidden_dim]
                        q_emb = q_reps[i:i+1]
                    else:
                        q_emb = q_reps
            
                    scores = self.compute_similarity(q_emb, p_rep_single[0])
                    scores = scores / self.temperature
                    scores = scores.squeeze(0)
                    scores_batch.append(scores)
            
                    # del p_rep_single
                    # torch.cuda.empty_cache()
                scores = scores_batch

            return EncoderOutput(
                loss=None,
                scores=scores,
                query_input_ids=query['input_ids'],
                passage_input_ids=passage['input_ids'],
                length=torch.tensor(passage['input_ids'].size(1), device=query['input_ids'].device),
            )

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
    
    def save_pretrained(self, output_dir: str, state_dict=None, **kwargs):
        if state_dict is None:
            state_dict = self.model.state_dict()
        
        state_dict = {k: v.clone().cpu() for k, v in state_dict.items()}
        
        self.model.save_pretrained(output_dir, state_dict=state_dict, **kwargs)


class CompressiveEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf",
        num_hidden_layers: int = 8,
        dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[str] = None,
        attn_implementation: str = "flash_attention_2",
        window: int = 1024,
        encoder_max_length: int = 4096,
        comp_candidates: List[int] = [2, 4, 8, 16, 32],
        pretraining_down_scaling_method: str = "stride",
        seed: int = 42,
        use_safetensors: bool = False,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            # trust_remote_code=True,
            num_hidden_layers=num_hidden_layers,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            use_safetensors=use_safetensors,
        )
        self.window = window
        self.encoder_max_length = encoder_max_length
        self.comp_candidates = comp_candidates
        self.pretraining_down_scaling_method = pretraining_down_scaling_method
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def save(self, output_dir):
        self.model.save_pretrained(
            os.path.join(output_dir, "compressive_encoder"), safe_serialization=True
        )

    def forward(self, input_ids, attention_mask, encoder_indices: List[List[int]]):
        # * base forward
        output = self.model.forward(input_ids, attention_mask)
        output = output.last_hidden_state  # [B, S, H]
        # * select according to encoder_indices
        encoder_embeds = []
        for idx, _encoder_indices in enumerate(encoder_indices):
            encoder_embed = output[idx][_encoder_indices]  # [ENCODER_LEN, H]
            encoder_embeds.append(encoder_embed)
        encoder_embeds = torch.cat(encoder_embeds, dim=0).contiguous()  # [SUM(ENCODER_LEN), H]

        return encoder_embeds

    def clear_cache(self):
        self.idx = None
        self.input_ids_segments = None
        self.attention_mask_segments = None
        self.labels_segments = None
        self.encoder_embeds_segments = None

        self.window_loss = []
        self.window_valid_token_num = []

    def prepare(self, input_ids, attention_mask, labels):
        batch_size, seq_size = input_ids.shape
        assert batch_size == 1, "Two-Steam AR training's batch_size must be 1."

        # * prepare segment sizes
        segment_num = math.ceil(seq_size / self.window)
        comp_ratios = []
        segment_sizes = []
        comp_segment_sizes = []

        for _ in range(segment_num):
            comp_ratio = self.rng.choice(self.comp_candidates)
            segment_size = min(self.window, seq_size - sum(segment_sizes))
            comp_segment_size = (
                segment_size // comp_ratio
            )  # Since the last segment won't be compressed, we don't need to consider the case where the number is not divisible by comp_ratio

            comp_ratios.append(comp_ratio)
            segment_sizes.append(segment_size)
            comp_segment_sizes.append(comp_segment_size)

        comp_segment_sizes.pop()  # The last segment can be deleted directly (no compression needed)

        # * prepare compressive encoder segment sizes and indices
        encoder_indices = []
        _encoder_indices = []
        encoder_segement_sizes = []
        encoder_segement_size = 0

        for i in range(segment_num):
            if encoder_segement_size >= self.encoder_max_length or i == segment_num - 1:
                encoder_indices.append(_encoder_indices.copy())
                encoder_segement_sizes.append(encoder_segement_size)

                _encoder_indices.clear()
                encoder_segement_size = 0
            if i == segment_num - 1:
                break
            
            if self.pretraining_down_scaling_method == "stride":
                _encoder_indices += [
                    encoder_segement_size + comp_ratios[i] * (j + 1) - 1
                    for j in range(comp_segment_sizes[i])
                ]
            elif self.pretraining_down_scaling_method == "random":
                indices = torch.randperm(segment_sizes[i], device=input_ids.device)[:comp_segment_sizes[i]]
                indices = (indices + encoder_segement_size).tolist()
                _encoder_indices += indices
            else:
                raise ValueError(f"Unknown pretraining_down_scaling_method: {self.pretraining_down_scaling_method}")

            encoder_segement_size += segment_sizes[i]

        # * format compressive encoder inputs
        encoder_input_ids = input_ids[:, : sum(encoder_segement_sizes)].split(
            encoder_segement_sizes, dim=1
        )
        encoder_attention_mask = attention_mask[:, : sum(encoder_segement_sizes)].split(
            encoder_segement_sizes, dim=1
        )

        # * get compressive encoder embeds
        encoder_embeds = []
        for i in range(len(encoder_input_ids)):
            _encoder_embeds = self.forward(
                encoder_input_ids[i], encoder_attention_mask[i], encoder_indices[i : i + 1]
            )
            encoder_embeds.append(_encoder_embeds)
        encoder_embeds = torch.cat(encoder_embeds).contiguous()  # [SUM(ENCODER_LEN), H]

        self.idx = 1
        self.input_ids_segments = input_ids.split(segment_sizes, dim=1)
        self.attention_mask_segments = attention_mask.split(segment_sizes, dim=1)
        self.labels_segments = labels.split(segment_sizes, dim=1)
        self.encoder_embeds_segments = encoder_embeds.split(comp_segment_sizes, dim=0)

    @property
    def is_finished(self):
        return self.idx >= len(self.input_ids_segments)

    def step(self):
        # * compressive encoder embeds
        encoder_embeds = torch.cat(self.encoder_embeds_segments[: self.idx])  # [SUM(ENCODER_LEN), H]
        # * placeholder indices
        ph_indices = [[i for i in range(encoder_embeds.shape[0])]]
        # * input_ids
        input_ids = self.input_ids_segments[self.idx]
        input_ids_ph = input_ids.new_ones(1, encoder_embeds.shape[0])
        input_ids = torch.cat([input_ids_ph, input_ids], dim=1)  # [1, S]
        # * attention_mask
        attention_mask = input_ids.new_ones(input_ids.shape)  # [1, S]
        # * labels
        labels = self.labels_segments[self.idx]
        labels_ph = labels.new_full((1, encoder_embeds.shape[0]), -100)
        labels = torch.cat([labels_ph, labels], dim=1)
        # * index go forward
        self.idx += 1

        return input_ids, attention_mask, labels, encoder_embeds, ph_indices

    def update_loss(self, loss: torch.FloatTensor, valid_token_num: int):
        self.window_valid_token_num.append(valid_token_num)
        self.window_loss.append(loss)

    @property
    def sample_loss(self):
        sample_loss = 0
        sample_valid_token_num = 0
        for loss, valid_token_num in zip(self.window_loss, self.window_valid_token_num):
            if torch.isnan(loss):
                continue
            sample_loss += loss * valid_token_num
            sample_valid_token_num += valid_token_num

        return sample_loss / sample_valid_token_num


class CompressionRateAdapter(nn.Module):
    def __init__(
        self,
        attention_padding_value: int = 0,
        label_padding_value: int = -100,
        model_name_or_path: str = None,
        peft_model_name_or_path: str = None,
        normalized: bool = True,
        pooling_method: str = 'cls',
        temperature: float = 0.02,
        lora_tune: bool = False,
        attn_implementation: str = "flash_attention_2",
        accelerator: Optional[Accelerator] = None,
    ):
        super().__init__()
        self.attention_padding_value = attention_padding_value
        self.label_padding_value = label_padding_value

        if model_name_or_path:
            self.embedding_model = BiEncoderModel(
                model_name=model_name_or_path,
                peft_model_name=peft_model_name_or_path,
                normalized=normalized,
                pooling_method=pooling_method,
                temperature=temperature,
                lora_tune=lora_tune,
                attn_implementation=attn_implementation,
                accelerator=accelerator
            )
        else:
            self.embedding_model = None

    def process_model_inputs(self, input_ids, ph_indices, labels, tokenizer):
        # * get attention mask
        max_len = get_max_length_in_nested_lists(input_ids)
        attention_mask = get_attention_mask_from_nested_lists(input_ids)

        # * get new ph_indices since padding side is left
        ph_indices = [
            [idx + max_len - len(input_ids[i]) for idx in ph_indices[i]]
            for i in range(len(ph_indices))
        ]
        if sum([len(x) for x in ph_indices]) == 0:
            ph_indices = None

        # * pad
        input_ids = pad_nested_lists(
            input_ids, max_len, tokenizer.pad_token_id, "left"
        )
        attention_mask = pad_nested_lists(
            attention_mask, max_len, self.attention_padding_value, "left"
        )
        if labels:
            labels = pad_nested_lists(labels, max_len, self.label_padding_value, "left")

        return input_ids, attention_mask, ph_indices, labels
    
    def process_encoder_inputs(self, encoder_input_ids, encoder_indices, tokenizer):
        # * 3D -> 2D
        encoder_input_ids = sum(encoder_input_ids, [])  # List[List[int]]
        encoder_indices = sum(encoder_indices, [])  # List[List[int]]
        # * filter empty item
        new_encoder_input_ids = []
        new_encoder_indices = []
        for i in range(len(encoder_input_ids)):
            if len(encoder_indices[i]) != 0:
                new_encoder_input_ids.append(encoder_input_ids[i])
                new_encoder_indices.append(encoder_indices[i])
        encoder_input_ids = new_encoder_input_ids
        encoder_indices = new_encoder_indices

        if len(encoder_input_ids) == 0:
            return [], [], None

        # * get attention mask and pad
        max_len = get_max_length_in_nested_lists(encoder_input_ids)
        encoder_attention_mask = get_attention_mask_from_nested_lists(encoder_input_ids)

        encoder_indices = [
            [idx + max_len - len(encoder_input_ids[i]) for idx in encoder_indices[i]]
            for i in range(len(encoder_indices))
        ]

        encoder_input_ids = pad_nested_lists(
            encoder_input_ids, max_len, tokenizer.pad_token_id, "left",
        )
        encoder_attention_mask = pad_nested_lists(
            encoder_attention_mask, max_len, self.attention_padding_value, "left"
        )

        return encoder_input_ids, encoder_attention_mask, encoder_indices

    @staticmethod
    def split_head_tail_context(input_ids_wo_context, input_ids_w_context):
        length_wo_context = len(input_ids_wo_context)
        head_input_ids = []
        for j in range(length_wo_context):
            if input_ids_wo_context[j] != input_ids_w_context[j]:
                break
            head_input_ids.append(input_ids_w_context[j])
        tail_input_ids = []
        for j in range(1, length_wo_context + 1):
            if input_ids_wo_context[-j] != input_ids_w_context[-j]:
                break
            tail_input_ids.append(input_ids_w_context[-j])
        tail_input_ids = tail_input_ids[::-1]
        context_input_ids = input_ids_w_context[len(head_input_ids):-len(tail_input_ids)]

        return head_input_ids, tail_input_ids, context_input_ids
    
    @staticmethod
    def get_encoder_input_ids(context_input_ids: List[int], encoder_max_length: int, bos_token_id: Optional[int] = None, eos_token_id: Optional[int] = None, task_instruction_ids: Optional[List[int]] = None,
    ) -> List[List[int]]:
        encoder_input_ids = []
        if task_instruction_ids is None:
            task_instruction_ids = []
        if bos_token_id and eos_token_id:
            step = encoder_max_length - 2 - len(task_instruction_ids)
        elif bos_token_id:
            step = encoder_max_length - 1 - len(task_instruction_ids)
        else:
            step = encoder_max_length - len(task_instruction_ids)
    
        for i in range(0, len(context_input_ids), step):
            if bos_token_id and eos_token_id:
                encoder_input_ids.append(
                    [bos_token_id] + task_instruction_ids + context_input_ids[i:i+step] + [eos_token_id]
                )
            elif bos_token_id:
                encoder_input_ids.append(
                    [bos_token_id] + task_instruction_ids + context_input_ids[i:i+step]
                )
            else:
                encoder_input_ids.append(
                    task_instruction_ids + context_input_ids[i:i+step]
                )
        return encoder_input_ids
    
    @staticmethod
    def get_encoder_indices(encoder_input_ids: List[List[int]], comp_ratio: int, method="stride", task_instruction_ids=None
    ) -> List[List[int]]:
        assert method in ["stride", "random", "terminal"], "Down scalng method is error. Make sure method in `['stride', 'random', 'terminal']`."
        if task_instruction_ids is None:
            task_instruction_ids = []
        encoder_indices = []
        for _encoder_input_ids in encoder_input_ids:
            _encoder_indices = list(range(comp_ratio - 1 + len(task_instruction_ids), len(_encoder_input_ids), comp_ratio))
            if len(_encoder_input_ids) % comp_ratio != 0:
                _encoder_indices.append(len(_encoder_input_ids) - 1)
            if comp_ratio == 1:
                _encoder_indices = _encoder_indices[1:]
            encoder_indices.append(_encoder_indices)

        if method == "stride":
            pass
        elif method == "random":
            new_encoder_indices = []
            for i in range(len(encoder_indices)):
                num = len(encoder_indices[i])
                _encoder_indices = random.sample(range(len(task_instruction_ids), len(encoder_input_ids[i])), num)
                _encoder_indices = sorted(_encoder_indices)
                new_encoder_indices.append(_encoder_indices)
            encoder_indices = new_encoder_indices
        elif method == "terminal":
            new_encoder_indices = []
            for i in range(len(encoder_indices)):
                num = len(encoder_indices[i])
                _encoder_indices = list(range(len(task_instruction_ids), len(encoder_input_ids[i])))[-num:]
                new_encoder_indices.append(_encoder_indices)
            encoder_indices = new_encoder_indices

        return encoder_indices

    def get_token_level_weighted_encoder_indices(self, encoder_input_ids: List[List[int]], high_comp_ratio: int, selected_token_indices: List[List[int]], method="stride", task_instruction_ids=None,
    ) -> List[List[int]]:
        assert method in ["stride", "random", "terminal"], "Down scalng method is error. Make sure method in `['stride', 'random', 'terminal']`."
        encoder_indices = []
        if task_instruction_ids is None:
            task_instruction_ids = []
        # * insert selected_token_indices into uniform indices
        for idx, _encoder_input_ids in enumerate(encoder_input_ids):
            _encoder_indices = list(range(high_comp_ratio - 1 + len(task_instruction_ids), len(_encoder_input_ids), high_comp_ratio))
            if len(_encoder_input_ids) % high_comp_ratio != 0:
                _encoder_indices.append(len(_encoder_input_ids) - 1)
            if high_comp_ratio == 1:
                _encoder_indices = _encoder_indices[1:]
            if selected_token_indices is not None and len(selected_token_indices) >= idx + 1:
                _encoder_indices_set = set(_encoder_indices)
                for token_index in selected_token_indices[idx]:
                    _encoder_indices_set.add(token_index)
                _encoder_indices = list(_encoder_indices_set)
                _encoder_indices = sorted(_encoder_indices)
            encoder_indices.append(_encoder_indices)

        return encoder_indices
    
    @staticmethod
    def build_placeholder_sequence(head, tail, encoder_indices):
        ph_indices_num = sum([len(x) for x in encoder_indices])
        ph_indices = [len(head) + j for j in range(ph_indices_num)]
        input_ids = head + [PH_TOKEN_ID] * ph_indices_num + tail
        return ph_indices_num, ph_indices, input_ids

    def uniform_allocation(self, 
        # data: List[dict],
        prompts: List[str],
        contexts: List[str],
        tokenizer: PreTrainedTokenizer,
        lm_max_length: int=4096,
        encoder_max_length: int=4096,
        comp_ratio: int=2,
        down_scaling_method: str="stride",
        task_instruction: str=None,
        apply_chat_template: bool=True,
    ):
        outputs = {
            "input_ids": [],
            "encoder_input_ids": [],
            "ph_indices": [],
            "encoder_indices": [],
            "length": [],
        }
        for prompt, context in zip(prompts, contexts):
            # * tokenize prompt without context, and then locate the position of the context_token_id
            if apply_chat_template:
                input_ids_wo_context = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                input_ids_wo_context = tokenizer(prompt, add_special_tokens=True)["input_ids"]

            # * tokenize prompt, and then split input_ids into 3 parts
            prompt_w_context = prompt.replace(CONTEXT_TAG, context)
            if apply_chat_template:
                input_ids_w_context = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_w_context}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                input_ids_w_context = tokenizer(prompt_w_context, add_special_tokens=True)["input_ids"]

            # * split input_ids into 3 parts
            head_input_ids, tail_input_ids, context_input_ids = self.split_head_tail_context(input_ids_wo_context, input_ids_w_context)

            # * truncate too long context
            special_token_num = 1 # do not involve eos token
            max_encoder_token_num = (lm_max_length - len(head_input_ids) - len(tail_input_ids)) * comp_ratio
            max_encoder_token_num -= math.ceil(max_encoder_token_num / encoder_max_length) * special_token_num
            if len(context_input_ids) > max_encoder_token_num:
                half = max_encoder_token_num // 2
                context_input_ids = context_input_ids[:half] + context_input_ids[-half:]

            if task_instruction:
                task_instruction_ids = tokenizer.encode(task_instruction, add_special_tokens=False)
            else:
                task_instruction_ids = None

            # * encoder_input_ids
            encoder_input_ids = self.get_encoder_input_ids(context_input_ids, encoder_max_length, bos_token_id=tokenizer.bos_token_id, task_instruction_ids=task_instruction_ids)

            # * encoder_indices
            encoder_indices = self.get_encoder_indices(encoder_input_ids, comp_ratio, down_scaling_method, task_instruction_ids)

            # * input_ids and ph_indices
            ph_indices_num, ph_indices, input_ids = self.build_placeholder_sequence(head_input_ids, tail_input_ids, encoder_indices)
            
            # * format
            outputs["input_ids"].append(input_ids)
            outputs["encoder_input_ids"].append(encoder_input_ids)
            outputs["ph_indices"].append(ph_indices)
            outputs["encoder_indices"].append(encoder_indices)
            outputs["length"].append(len(input_ids))

        input_ids = outputs["input_ids"]
        encoder_input_ids = outputs["encoder_input_ids"]
        ph_indices = outputs["ph_indices"]
        encoder_indices = outputs["encoder_indices"]
        labels = None

        # * process model inputs
        input_ids, attention_mask, ph_indices, labels = self.process_model_inputs(
            input_ids, ph_indices, labels, tokenizer
        )

        # * process compressive encoder input
        encoder_input_ids, encoder_attention_mask, encoder_indices = self.process_encoder_inputs(
            encoder_input_ids, encoder_indices, tokenizer
        )

        # * to torch tensor
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        encoder_input_ids = torch.tensor(encoder_input_ids)
        encoder_attention_mask = torch.tensor(encoder_attention_mask)
        labels = torch.tensor(labels) if labels else None
        
        # * format
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ph_indices": ph_indices,
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "encoder_indices": encoder_indices,
            "labels": labels,
        }
        
    def token_level_adaptation(self,
        prompts: List[str],
        contexts: List[str],
        tokenizer: PreTrainedTokenizer,
        queries: Optional[str]=None,
        lm_max_length: int=4096,
        encoder_max_length: int=4096,
        comp_ratio: int=2,
        down_scaling_method: str="stride",
        task_instruction: str=None,
        apply_chat_template: bool=True,
        query_max_length: int=32,
        passage_max_length: int=80000,
        context_proportion: float=0.1, # the proportion of the importance context in the original context
        low_comp_ratio: int=1, # the compression ratio for important context
    ):
        outputs = {
            "input_ids": [],
            "encoder_input_ids": [],
            "ph_indices": [],
            "encoder_indices": [],
            "length": [],
        }
        all_p_inputs = {"input_ids": [], "attention_mask": []}       
        all_head_ids = []
        all_tail_ids = []
        all_context_ids = []      
        all_length = []  

        for prompt, context in zip(prompts, contexts):
            # * tokenize prompt without context, and then locate the position of the context_token_id
            if apply_chat_template:
                input_ids_wo_context = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                input_ids_wo_context = tokenizer(prompt, add_special_tokens=True)["input_ids"]

            # * tokenize prompt, and then split input_ids into 3 parts
            prompt_w_context = prompt.replace(CONTEXT_TAG, context)
            if apply_chat_template:
                input_ids_w_context = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_w_context}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                input_ids_w_context = tokenizer(prompt_w_context, add_special_tokens=True)["input_ids"]

            # * split input_ids into 3 parts
            head_input_ids, tail_input_ids, context_input_ids = self.split_head_tail_context(input_ids_wo_context, input_ids_w_context)

            # * truncate too long context
            special_token_num = 1 # do not involve eos token
            max_encoder_token_num = (lm_max_length - len(head_input_ids) - len(tail_input_ids)) * comp_ratio
            max_encoder_token_num -= math.ceil(max_encoder_token_num / encoder_max_length) * special_token_num
            if len(context_input_ids) > max_encoder_token_num:
                half = max_encoder_token_num // 2
                context_input_ids = context_input_ids[:half] + context_input_ids[-half:]

            all_head_ids.append(head_input_ids)
            all_tail_ids.append(tail_input_ids)
            all_context_ids.append(context_input_ids)

            # * encoder_input_ids
            encoder_input_ids_wo_task_instruction = self.get_encoder_input_ids(context_input_ids, encoder_max_length, bos_token_id=tokenizer.bos_token_id, task_instruction_ids=None)

            # * merge multiple encoder_input_ids
            encoder_input_ids_merged = []
            for j in range(len(encoder_input_ids_wo_task_instruction)):
                encoder_input_ids_merged.extend(encoder_input_ids_wo_task_instruction[j])

            # * get important token indices
            all_p_inputs["input_ids"].append(encoder_input_ids_merged)
            all_p_inputs["attention_mask"].append([1] * len(encoder_input_ids_merged))
            all_length.append(len(encoder_input_ids_merged))

        p_batch = {} 
        max_len = get_max_length_in_nested_lists(all_p_inputs["input_ids"])
        p_batch["input_ids"] = torch.tensor(
            pad_nested_lists(all_p_inputs["input_ids"], max_len, tokenizer.pad_token_id, "left")
        ).to(self.embedding_model.model.device)
        p_batch["attention_mask"] = torch.tensor(
            pad_nested_lists(all_p_inputs["attention_mask"], max_len, self.attention_padding_value, "left")
        ).to(self.embedding_model.model.device)

        q_batch = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=query_max_length,
            return_tensors="pt",
            padding_side="left",
        ).to(self.embedding_model.model.device)

        output = self.embedding_model(q_batch, p_batch)
        all_scores = output.scores

        for i in range(len(prompts)):
            mask = p_batch["attention_mask"][i]
            valid_length = mask.sum().item()
            scores = all_scores[i][-valid_length:]
            target_length = math.floor(context_proportion * all_length[i])

            important_token_indices = (-scores).argsort()[:target_length].tolist()
            important_token_indices = convert_token_indices(len(scores), important_token_indices, encoder_max_length)

            # * adapt compression ratio
            if task_instruction:
                task_instruction_ids = tokenizer.encode(task_instruction, add_special_tokens=False)
            else:
                task_instruction_ids = None

            encoder_input_ids = self.get_encoder_input_ids(all_context_ids[i], encoder_max_length, bos_token_id=tokenizer.bos_token_id, task_instruction_ids=task_instruction_ids)
            
            # * convert indices when applying task prompt
            if task_instruction_ids:
                indices_temp = []
                for idx, _important_token_indices in enumerate(important_token_indices):
                    if len(indices_temp) < idx + 1:
                        indices_temp.append([])
                    for x in _important_token_indices:
                        if x + len(task_instruction_ids) < encoder_max_length:
                            indices_temp[idx].append(x + len(task_instruction_ids))
                        else:
                            if len(indices_temp) < idx + 2:
                                indices_temp.append([])
                            indices_temp[idx + 1].append(x + len(task_instruction_ids) + special_token_num - encoder_max_length)
                important_token_indices = indices_temp

            # * select token_ids based on low comp ratio
            len_encoder_input_ids = 0
            for _encoder_input_ids in encoder_input_ids:
                len_encoder_input_ids += len(_encoder_input_ids) - len(task_instruction_ids) - special_token_num

            low_ratio_index_length = 0 
            selected_token_indices = []

            for _importance_token_indices in important_token_indices:
                temp = []
                for idx, token_index in enumerate(_importance_token_indices):
                    if idx % low_comp_ratio == 0:
                        temp.append(token_index)
                if len(_importance_token_indices) % low_comp_ratio != 0:
                    temp.append(_importance_token_indices[-1])
                low_ratio_index_length += len(temp)
                selected_token_indices.append(temp)

            # * Based on the length of imporatance sentences after compress, calculate proper ratio for remain context
            len_for_encoder = math.ceil(len(all_context_ids[i]) / comp_ratio)
            remain_length = len_for_encoder - low_ratio_index_length 

            if remain_length <= 0:
                high_comp_ratio = 0
            else:
                high_comp_ratio = math.ceil((len_encoder_input_ids - low_ratio_index_length) / remain_length)

            # * encoder_indices
            if low_ratio_index_length > 0 and high_comp_ratio > 0 and high_comp_ratio < 400:
                encoder_indices = self.get_token_level_weighted_encoder_indices(encoder_input_ids, high_comp_ratio, selected_token_indices, down_scaling_method, task_instruction_ids)

                # make sure that lm_length is less than len_for_encoder
                lm_length = sum([len(x) for x in encoder_indices])
                while lm_length > len_for_encoder:
                    high_comp_ratio += 1
                    encoder_indices = self.get_token_level_weighted_encoder_indices(encoder_input_ids, high_comp_ratio, selected_token_indices, down_scaling_method, task_instruction_ids)
                    lm_length = sum([len(x) for x in encoder_indices])
                    if high_comp_ratio > 400:
                        encoder_indices = self.get_encoder_indices(encoder_input_ids, comp_ratio, down_scaling_method, task_instruction_ids)
                        break
            else:
                encoder_indices = self.get_encoder_indices(encoder_input_ids, comp_ratio, down_scaling_method, task_instruction_ids)

            # * input_ids and ph_indices
            ph_indices_num, ph_indices, input_ids = self.build_placeholder_sequence(all_head_ids[i], all_tail_ids[i], encoder_indices)

            # * format
            outputs["input_ids"].append(input_ids)
            outputs["encoder_input_ids"].append(encoder_input_ids)
            outputs["ph_indices"].append(ph_indices)
            outputs["encoder_indices"].append(encoder_indices)

        input_ids = outputs["input_ids"]
        encoder_input_ids = outputs["encoder_input_ids"]
        ph_indices = outputs["ph_indices"]
        encoder_indices = outputs["encoder_indices"]
        labels = None

        # * process model inputs
        input_ids, attention_mask, ph_indices, labels = self.process_model_inputs(
            input_ids, ph_indices, labels, tokenizer
        )

        # * process compressive encoder input
        encoder_input_ids, encoder_attention_mask, encoder_indices = self.process_encoder_inputs(
            encoder_input_ids, encoder_indices, tokenizer
        )

        # * to torch tensor
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        encoder_input_ids = torch.tensor(encoder_input_ids)
        encoder_attention_mask = torch.tensor(encoder_attention_mask)
        labels = torch.tensor(labels) if labels else None
        
        # * format
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ph_indices": ph_indices,
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "encoder_indices": encoder_indices,
            "labels": labels,
        }

@dataclass
class CausalLMOutputForWindow(CausalLMOutput):
    window_loss: Optional[List[torch.FloatTensor]] = None
    window_valid_token_num: Optional[List[int]] = None


class TacZipConfig(PretrainedConfig):
    model_type = "TacZip"

    def __init__(
        self,
        language_model_name_or_path: str = None,
        encoder_name_or_path: str = None,
        embedding_model_name_or_path: str = None, # text embedding model
        num_hidden_layers: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.language_model_name_or_path = language_model_name_or_path # language model name
        self.encoder_name_or_path = encoder_name_or_path # compressive encoder model name
        self.embedding_model_name_or_path = embedding_model_name_or_path # embedding model name
        self.num_hidden_layers = num_hidden_layers # the number of encoder layers
        

class TacZip(PreTrainedModel):
    config_class = TacZipConfig
    base_model_prefix = "taczip"

    def __init__(
        self, 
        config: TacZipConfig,
        window_mode: bool = False, # enable two stream auto regressive training
        window: int = 1024, # window size when perfroming two stream auto regressive training
        lm_max_length: int = 4096, # maximum token length for language model inputs
        encoder_max_length: int = 4096, # maxinum token length for compressive encoder inputs
        comp_candidates: Optional[List[int]] = None, # compression ratio candidates for training
        pretraining_down_scaling_method: str = "stride", # down scaling method used during pretraing
        embedding_peft_model_name_or_path: Optional[str] = None, # PEFT fine-tuned text embedding model path
        normalized: bool = True, # whether to apply normalization to embedding model outputs last hidden states
        pooling_method: str = 'cls', # pooling method for query embeddings
        temperature: float = 0.02, # embedding model's temperature
        dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[str] = None,
        attn_implementation: str = "flash_attention_2",
        accelerator: Optional[Accelerator] = None,
        seed: int = 42,
        use_safetensors: bool = False,
    ):
        super().__init__(config)
        # * init model
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.language_model_name_or_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )

        if config.encoder_name_or_path:
            # * set compressive encoder
            self.compressive_encoder = CompressiveEncoder(
                config.encoder_name_or_path, 
                config.num_hidden_layers, 
                dtype, 
                device_map, 
                attn_implementation,
                window,
                encoder_max_length,
                comp_candidates,
                pretraining_down_scaling_method,
                seed,
                use_safetensors=use_safetensors,
            )
        else:
            self.compressive_encoder = None

        # * set compression-rate adapter
        self.compression_rate_adapter = CompressionRateAdapter(
            model_name_or_path=config.embedding_model_name_or_path,
            peft_model_name_or_path=embedding_peft_model_name_or_path,
            normalized=normalized,
            pooling_method=pooling_method,
            temperature=temperature,
            attn_implementation=attn_implementation,
            accelerator=accelerator,
        )

        # * set other parameters
        self.window_mode = window_mode
        self.lm_max_length = lm_max_length

        # * freeze model
        self.freeze_model()

        # * set accelerator
        self.accelerator = accelerator
        if device_map is None:
            if self.accelerator is not None:
                device = self.accelerator.device
            else:
                device = torch.device("cpu")
            self.language_model.to(device)
            if self.compressive_encoder:
                self.compressive_encoder.to(device)
        
            if self.compression_rate_adapter.embedding_model:
                self.compression_rate_adapter.embedding_model.to(device)

    @property
    def device(self):
        if self.accelerator:
            return self.accelerator.device
        else:
            return self.language_model.device

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.compressive_encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs
        )

    def save(self, output_dir):
        self.compressive_encoder.save(output_dir)

    def _two_stream_ar_forward(self, input_ids, attention_mask, labels):
        self.compressive_encoder.clear_cache()
        self.compressive_encoder.prepare(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        while not self.compressive_encoder.is_finished:
            (
                input_ids,
                attention_mask,
                labels,
                encoder_embeds,
                ph_indices,
            ) = self.compressive_encoder.step()
            inputs_embeds = self.prepare_inputs_embeds(input_ids, encoder_embeds, ph_indices)
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            valid_token_num = (labels[:, 1:] != -100).sum()
            self.compressive_encoder.update_loss(outputs.loss, valid_token_num)

        window_loss = self.compressive_encoder.window_loss
        window_valid_token_num = self.compressive_encoder.window_valid_token_num
        sample_loss = self.compressive_encoder.sample_loss

        return CausalLMOutputForWindow(
            loss=sample_loss,
            window_loss=window_loss,
            window_valid_token_num=window_valid_token_num,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        encoder_indices: Optional[List[List[int]]] = None,
        ph_indices: Optional[List[List[int]]] = None,
    ):
        if self.compressive_encoder:
            if self.window_mode:
                return self._two_stream_ar_forward(input_ids, attention_mask, labels)
            else:
                if ph_indices and encoder_indices:
                    encoder_embeds = self.get_encoder_embeds(
                        encoder_input_ids, encoder_attention_mask, encoder_indices
                    )
                    inputs_embeds = self.prepare_inputs_embeds(
                        input_ids, encoder_embeds, ph_indices
                    )
                    return self.language_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                elif ph_indices is None and encoder_indices is None:
                    return self.language_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                else:
                    raise ValueError(
                        "Arguments `ph_indices` and `encoder_indices` must be all `None` or not."
                    )
        else:
            return self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

    def get_encoder_embeds(self, encoder_input_ids, encoder_attention_mask, encoder_indices):
        encoder_embeds = []
        for idx, _encoder_indices in enumerate(encoder_indices):
            if not _encoder_indices:
                continue
            _encoder_embeds = self.compressive_encoder(
                encoder_input_ids[[idx]], encoder_attention_mask[[idx]], [_encoder_indices]
            )  # [ENCODER_LEN, H]
            encoder_embeds.append(_encoder_embeds)
        encoder_embeds = torch.cat(encoder_embeds).contiguous()  # [SUM(ENCODER_LEN), H]

        return encoder_embeds

    def prepare_inputs_embeds(self, input_ids, encoder_embeds, ph_indices: List[List[int]]):
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        idx = 0
        for i, _ph_indices in enumerate(ph_indices):
            if not _ph_indices:
                continue
            inputs_embeds[i][_ph_indices] = encoder_embeds[idx : idx + len(_ph_indices)]
            idx += len(_ph_indices)

        return inputs_embeds

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        attention_mask,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        encoder_indices: Optional[List[List[int]]] = None,
        ph_indices: Optional[List[List[int]]] = None,
        **gen_kwargs,
    ):
        self.eval()

        if self.compressive_encoder:
            if ph_indices and encoder_indices:
                encoder_embeds = self.get_encoder_embeds(
                    encoder_input_ids, encoder_attention_mask, encoder_indices
                )
                inputs_embeds = self.prepare_inputs_embeds(
                    input_ids, encoder_embeds, ph_indices
                )
                return self.language_model.generate(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )
            elif ph_indices is None and encoder_indices is None:
                return self.language_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            else:
                raise ValueError(
                    "Arguments `ph_indices` and `encoder_indices` must be all `None` or not."
                )
        else:
            return self.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        

    def freeze_model(self):
        for _, param in self.language_model.named_parameters():
            param.requires_grad = False

    def _move_to_device(self, inputs):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        return inputs

# * Utilities

def get_max_length_in_nested_lists(lst):
    if isinstance(lst[0], list):
        lengths = []
        for elem in lst:
            length = get_max_length_in_nested_lists(elem)
            lengths.append(length)
        max_length = max(lengths)
        return max_length
    else:
        return len(lst)
    

def get_attention_mask_from_nested_lists(lst):
    if isinstance(lst[0], list):
        attention_mask = []
        for elem in lst:
            mask = get_attention_mask_from_nested_lists(elem)
            attention_mask.append(mask)
        return attention_mask
    else:
        return [1] * len(lst)


def pad_nested_lists(lst, max_length, padding_value, padding_side="right"):
    if isinstance(lst, list) and len(lst) and isinstance(lst[0], list):
        for i, elem in enumerate(lst):
            lst[i] = pad_nested_lists(elem, max_length, padding_value, padding_side)
        return lst
    elif isinstance(lst, list):
        if padding_side == "right":
            return lst + [padding_value for _ in range(max_length - len(lst))]
        else:
            return [padding_value for _ in range(max_length - len(lst))] + lst
    else:
        raise NotImplementedError(f"Unrecognized type {lst}")
    
    
def move_to_device(inputs, device):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs


def convert_token_indices(overall_length, importance_token_indices, encoder_max_length=4096):
    # convert importance token indices:
    # e.g. [[12, 23, 4096, 4098]] => [[12, 23], [0, 2]] 
    if overall_length <= encoder_max_length:
        return [importance_token_indices]
    token_indices = [[]]
    for _ in range(math.floor(overall_length / encoder_max_length)):
        token_indices.append([])
    for idx2 in importance_token_indices:
        idx1 = math.floor(idx2 / encoder_max_length)
        token_indices[idx1].append(idx2 - idx1 * encoder_max_length)

    return token_indices
