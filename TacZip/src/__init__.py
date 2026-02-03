import logging
from typing import Optional
from accelerate import Accelerator
from transformers import AutoTokenizer
from src.args import ModelArgs, LoraArgs
from src.data import CONTEXT_TAG
from src.utils import str_to_torch_dtype
from src.modeling_taczip import TacZip, TacZipConfig

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(
    model_args: ModelArgs,
    lora_args: LoraArgs = None,
    accelerator: Optional[Accelerator] = None,
    return_tokenizer_only: bool = False,
):
    # * First load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.language_model_name_or_path,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # * If `return_tokenizer_only` is True, we can return immediately.
    if return_tokenizer_only:
        logger.info("Only return tokenizer.")
        return tokenizer
    
    config = TacZipConfig(
        model_args.language_model_name_or_path,
        model_args.encoder_name_or_path,
        model_args.encoder_num_hidden_layers,
    )

    model = TacZip(
        config=config,
        embedding_model_name_or_path=model_args.embedding_model_name_or_path,
        window_mode=model_args.window_mode,
        window=model_args.window,
        lm_max_length=model_args.lm_max_length,
        encoder_max_length=model_args.encoder_max_length,
        comp_candidates=model_args.comp_candidates,
        pretraining_down_scaling_method=model_args.pretraining_down_scaling_method,
        dtype=str_to_torch_dtype(model_args.dtype),
        device_map=model_args.device_map,
        attn_implementation=model_args.attn_implementation,
        use_safetensors=model_args.use_safetensors,
        accelerator=accelerator,
    )

    if not model_args.device_map and accelerator:
        model.to(accelerator.device)
    
    if not model_args.window_mode:
        logger.info(
            "`window_mode` is False, so add `CONTEXT_TAG` as speical token."
        )
        tokenizer.add_tokens([CONTEXT_TAG], special_tokens=True)

    tokenizer.padding_side = "left"

    return model, tokenizer