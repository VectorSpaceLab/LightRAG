import random
import torch
import datasets
import os
import math
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from transformers import BatchEncoding, DataCollatorWithPadding
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from src.modeling_taczip import CompressionRateAdapter

logger = logging.get_logger(__name__)
PH_TOKEN_ID = 100
INPUT_TAG = "[INPUT_RmehNsY1]"
CONTEXT_TAG = "[CONTEXT_RmehNsY1]"


# * Main Data Class
class Data:
    @staticmethod
    def encode_with_labels(tokenizer, messages):
        user_messages = messages[:-1] 
    
        full_input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    
        user_input_ids = tokenizer.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True, 
        )
    
        labels = [-100] * len(full_input_ids)
    
        user_len = len(user_input_ids)
        for i in range(user_len, len(full_input_ids)):
            labels[i] = full_input_ids[i]
        
        return {
            "input_ids": full_input_ids,
            "labels": labels
        }

    @staticmethod
    def encode_pretraining_data(
        data: List[dict],
        indices: List[int],
        tokenizer: PreTrainedTokenizer,
        min_length: Optional[int]=None,
        max_length: Optional[int]=None,
    ):
        outputs = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
            "length": [],
            "index": [],
        }

        for i, text in enumerate(data["text"]):
            encoded = tokenizer(text, return_tensors=None)
            input_ids = encoded["input_ids"]
            labels = input_ids.copy()
            if len(input_ids) <= min_length or len(input_ids) >= max_length:
                continue
            attention_mask = [1 for _ in range(len(input_ids))]

            # * format
            outputs["input_ids"].append(input_ids)
            outputs["labels"].append(labels)
            outputs["attention_mask"].append(attention_mask)
            outputs["length"].append(len(input_ids))
            outputs["index"].append(indices[i])

        return outputs
    
    @staticmethod
    def encode_instruction_tuning_data(
        data: List[dict],
        indices: List[int],
        tokenizer: PreTrainedTokenizer,
        lm_max_length: int,
        encoder_max_length: int,
        comp_candidates: List[int],
        down_scaling_method: str="stride",
        task_instruction: str=None,
        min_length: Optional[int]=None,
        max_length: Optional[int]=None,
    ):
        outputs = {
            "input_ids": [],
            "encoder_input_ids": [],
            "ph_indices": [],
            "encoder_indices": [],
            "length": [],
            "index": [],
            "labels": [],
        }

        for i, conversations in enumerate(data["conversations"]):
            # * select compression ratio
            comp_ratio = random.choice(comp_candidates)  

            # * tokenize prompt without context, and then locate the position of the context_token_id
            prompt = conversations[0]["prompt"]
            messages = [{"role": "user", "content": prompt}] + conversations[1:]
            encoded_wo_context = Data.encode_with_labels(tokenizer, messages)

            # * tokenize prompt, and then split input_ids into 3 parts
            prompt_w_context = prompt.replace(CONTEXT_TAG, conversations[0]["context"])
            messages = [{"role": "user", "content": prompt_w_context}] + conversations[1:]
            encoded_w_context = Data.encode_with_labels(tokenizer, messages)

            length_w_context = len(encoded_w_context["input_ids"])

            # * split input_ids into 3 parts
            head_input_ids, tail_input_ids, context_input_ids = CompressionRateAdapter.split_head_tail_context(encoded_wo_context["input_ids"], encoded_w_context["input_ids"])

            # skip data that not fall in between min_length and max_length
            if length_w_context >= max_length or length_w_context <= min_length:
                continue

            # * truncate context_input_ids
            context_length = len(context_input_ids)
            remain_length = (
                lm_max_length - (length_w_context - context_length)
            ) * comp_ratio
            if remain_length < context_length:
                half = remain_length // 2
                context_input_ids = context_input_ids[:half] + context_input_ids[-half:]

            if task_instruction:
                task_instruction_ids = tokenizer.encode(task_instruction, add_special_tokens=False)
            else:
                task_instruction_ids = None

            # * encoder_input_ids
            encoder_input_ids = CompressionRateAdapter.get_encoder_input_ids(context_input_ids, encoder_max_length, tokenizer.bos_token_id, task_instruction_ids=task_instruction_ids)
            
            # * encoder_indices
            encoder_indices = CompressionRateAdapter.get_encoder_indices(encoder_input_ids, comp_ratio, down_scaling_method, task_instruction_ids)
            
            # * input_ids and ph_indices
            ph_indices_num, ph_indices, input_ids = CompressionRateAdapter.build_placeholder_sequence(head_input_ids, tail_input_ids, encoder_indices)

            # * format
            outputs["input_ids"].append(input_ids)
            outputs["encoder_input_ids"].append(encoder_input_ids)
            outputs["ph_indices"].append(ph_indices)
            outputs["encoder_indices"].append(encoder_indices)

            head_labels = encoded_w_context["labels"][:len(head_input_ids)]
            tail_labels = encoded_w_context["labels"][-len(tail_input_ids):]

            labels = head_labels + [-100] * ph_indices_num + tail_labels
            outputs["labels"].append(labels)
            outputs["length"].append(len(input_ids))
            outputs["index"].append(indices[i])

        return outputs

# * Colloator
@dataclass
class DefaultDataCollator:
    """
    Data collator that can:
    1. Dynamically pad all inputs received. The inputs must be dict of lists.
    2. Add position_ids based on attention_mask if required.
    """
    tokenizer: PreTrainedTokenizer
    attention_padding_value: int = 0
    label_padding_value: int = -100

    keys_to_tensorize = {
        "input_ids",
        "attention_mask",
        "labels",
        "position_ids",
        "token_type_ids",
        "length",
        "depth",
        "index",
    }

    def __call__(self, batch_elem: List) -> Dict[str, Any]:
        first_elem = batch_elem[0]
        return_batch = {}

        for key, value in first_elem.items():
            # HACK: any key containing attention_mask must be attention_mask
            # important to assign different pad token for different types of inputs
            if "attention_mask" in key:
                pad_token_id = self.attention_padding_value
            elif "label" in key:
                pad_token_id = self.label_padding_value
            else:
                pad_token_id = self.tokenizer.pad_token_id

            batch_value = [elem[key] for elem in batch_elem]
            # pad all lists and nested lists
            if isinstance(value, list) and key in self.keys_to_tensorize:
                max_length = get_max_length_in_nested_lists(batch_value)
                batch_value = pad_nested_lists(
                    batch_value, max_length, pad_token_id, self.tokenizer.padding_side
                )

            if key in self.keys_to_tensorize:
                return_batch[key] = torch.tensor(batch_value)
            else:
                # handle strings and None
                return_batch[key] = batch_value
        return return_batch


@dataclass
class SFTDataCollator:
    tokenizer: PreTrainedTokenizer
    attention_padding_value: int = 0
    label_padding_value: int = -100

    def __call__(self, batch_elem: List) -> Dict[str, Any]:
        # * extract data from features, and format them from dict to list
        input_ids = [f["input_ids"] for f in batch_elem]  # List[List[int]]
        ph_indices = [f["ph_indices"] for f in batch_elem]  # List[List[int]]
        encoder_input_ids = [f["encoder_input_ids"] for f in batch_elem]  # List[List[List[int]]]
        encoder_indices = [f["encoder_indices"] for f in batch_elem]  # List[List[List[int]]]
        labels = (
            [f["labels"] for f in batch_elem] if "labels" in batch_elem[0] else None
        )  # List[List[int]]

        # * process model inputs
        input_ids, attention_mask, ph_indices, labels = self.process_model_inputs(
            input_ids, ph_indices, labels
        )

        # * process compressive encoder input
        encoder_input_ids, encoder_attention_mask, encoder_indices = self.process_encoder_inputs(
            encoder_input_ids, encoder_indices
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

    def process_model_inputs(self, input_ids, ph_indices, labels):
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
            input_ids, max_len, self.tokenizer.pad_token_id, "left"
        )
        attention_mask = pad_nested_lists(
            attention_mask, max_len, self.attention_padding_value, "left"
        )
        if labels:
            labels = pad_nested_lists(labels, max_len, self.label_padding_value, "left")

        return input_ids, attention_mask, ph_indices, labels

    def process_encoder_inputs(self, encoder_input_ids, encoder_indices):
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
            encoder_input_ids, max_len, self.tokenizer.pad_token_id, "left",
        )
        encoder_attention_mask = pad_nested_lists(
            encoder_attention_mask, max_len, self.attention_padding_value, "left"
        )

        return encoder_input_ids, encoder_attention_mask, encoder_indices


# alignment between without selective compression and with selective compression
def get_compressive_encoder_inputs(
    conversations: list,
    tokenizer: PreTrainedTokenizer,
    lm_max_length: int,
    encoder_max_length: int,
    comp_ratio: int=8,
):
    conversations = [
        conversations[0],
        {"role": "assistant", "content": None},
    ]

    # * tokenize prompt without context, and then locate the position of the context_token_id
    prompt = conversations[0]["prompt"]
    input_ids_wo_context = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    # * tokenize prompt, and then split input_ids into 3 parts
    prompt_w_context = prompt.replace(CONTEXT_TAG, conversations[0]["context"])
    input_ids_w_context = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_w_context}],
        tokenize=True,
        add_generation_prompt=True,
    )
    head_input_ids, tail_input_ids, context_input_ids = CompressionRateAdapter.split_head_tail_context(input_ids_wo_context, input_ids_w_context)
   
    # * truncate too long context
    max_encoder_token_num = (lm_max_length - len(head_input_ids) - len(tail_input_ids)) * comp_ratio

    if len(context_input_ids) > max_encoder_token_num:
        half = max_encoder_token_num // 2
        context_input_ids = context_input_ids[:half] + context_input_ids[-half:]

    # * encoder input_ids
    encoder_input_ids_wo_task_instruction = CompressionRateAdapter.get_encoder_input_ids(context_input_ids,encoder_max_length)
        
    # merge multiple encoder_input_ids
    encoder_input_ids_merged = [[]]
    for j in range(len(encoder_input_ids_wo_task_instruction)):
        encoder_input_ids_merged[0].extend(encoder_input_ids_wo_task_instruction[j])

    attention_mask = [[1] * len(encoder_input_ids_merged[0])]
        
    return encoder_input_ids_merged, attention_mask


@dataclass
class SequenceLevelEmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    sub_batch_size: int = -1

    def __call__(self, features, return_tensors='pt'):
        if return_tensors is None:
            return_tensors = self.return_tensors

        queries = []
        passages = []
        for e in features:
            queries.append(e[0])
            passages.extend(e[1])

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_collated = self.tokenizer.pad(
                queries,
                padding=self.padding,
                max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )

            p_collated = self.tokenizer.pad(
                passages,
                padding=self.padding,
                max_length=self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
        else:
            batch_size = self.sub_batch_size

            q_collated = []
            for i in range(0, len(queries), batch_size):
                start = i
                end = min(len(queries), i + batch_size)
                sub_features = queries[start:end]
                q_collated.append(self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=return_tensors,
                ))

            p_collated = []
            for i in range(0, len(passages), batch_size):
                start = i
                end = min(len(passages), i + batch_size)
                sub_features = passages[start: end]
                p_collated.append(self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=return_tensors,
                ))

        return {"query": q_collated, "passage": p_collated}


@dataclass
class TokenLevelEmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    tokenizer: PreTrainedTokenizer = None
    query_max_len: int = 32

    def __call__(self, features):
        query = [f[0] for f in features]
        conversations = [f[1] for f in features]
        gold_indices = [f[2] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(conversations[0], list):
            conversations = sum(conversations, [])
        if isinstance(gold_indices[0], list):
            gold_indices = sum(gold_indices, [])

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )

        input_ids, attention_mask = get_compressive_encoder_inputs(
            conversations,
            self.tokenizer,
            lm_max_length=4096,
            encoder_max_length=4096,
            comp_ratio=8,
        )
        
        p_collated = {}
        p_collated['input_ids'] = torch.tensor(input_ids)
        p_collated['attention_mask'] = torch.tensor(attention_mask)
        total_length = p_collated['input_ids'].size(1)
        tags = torch.zeros(total_length)

        try:
            tags[gold_indices[0]] = 1 
        except:
            print('gold_indices', gold_indices)
            print('total_length', total_length)
            exit()        
        gold_indices = torch.tensor(gold_indices)

        return {"query": q_collated, "passage": p_collated, "tags": tags, "gold_indices": gold_indices}
    

class SequenceLevelTrainDataset(Dataset):
    def __init__(
        self,
        data_files,
        train_group_size,
        query_max_len,
        passage_max_len,
        tokenizer: PreTrainedTokenizer
    ):
        dataset_list = []
        for data_file in data_files:
            dataset = datasets.load_dataset('json', data_files=data_file, split='train')
            dataset_list.append(dataset)

        dataset = datasets.concatenate_datasets(dataset_list)

        self.dataset = dataset
        self.total_len = len(self.dataset)
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.train_group_size = train_group_size
        self.passage_max_len = passage_max_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        query = self.dataset[item]['query']
        query_inputs = self.tokenizer(query,
                                      return_tensors=None,
                                      max_length=self.query_max_len,
                                      truncation=True,
                                      add_special_tokens=False)
        query_inputs['input_ids'] = query_inputs['input_ids']
        query_inputs['attention_mask'] = [1] * len(query_inputs['input_ids'])

        passages = []
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)

        if len(self.dataset[item]['neg']) < self.train_group_size - 1:
            num = math.ceil((self.train_group_size - 1) / len(list(set(self.dataset[item]['neg']))))
            negs = random.sample(self.dataset[item]['neg'] * num, self.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.train_group_size - 1)

        passages.extend(negs)

        passages_inputs = []
        for passage in passages:
            passage_inputs = self.tokenizer(passage,
                                            return_tensors=None,
                                            max_length=self.passage_max_len,
                                            truncation=True,
                                            add_special_tokens=False)
            passage_inputs['input_ids'] = passage_inputs['input_ids']
            passage_inputs['attention_mask'] = [1] * len(passage_inputs['input_ids'])
            passages_inputs.append(passage_inputs)

        return query_inputs, passages_inputs


class TokenLevelTrainDataset(Dataset):
    def __init__(
        self,
        data_files,
        query_instruction_for_retrieval = None,
    ):
        dataset_list = []
        for data_file in data_files:
            dataset = datasets.load_dataset("json", data_files=data_file, split="train")
            dataset_list.append(dataset)

        dataset = datasets.concatenate_datasets(dataset_list)
        self.dataset = dataset
        self.total_len = len(self.dataset)
        self.print_flag = True
        self.query_instruction_for_retrieval = query_instruction_for_retrieval

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        query = self.dataset[item]['query']
        query = query + ' </s>'
        if self.query_instruction_for_retrieval:
            query = self.query_instruction_for_retrieval + query
    
        conversations = self.dataset[item]['conversations']
        gold_indices = self.dataset[item]['gold_indices']

        return query, conversations, gold_indices


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
