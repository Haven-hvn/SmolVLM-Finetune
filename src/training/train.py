import os
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForVision2Seq, HfArgumentParser
from training.trainer import SmolVLMTrainer
from training.data import make_supervised_data_module
from training.params import DataArguments, ModelArguments, TrainingArguments
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
import warnings

# Image handling imports
from PIL import Image, ImageFile

# AVIF support initialization
try:
    from pillow_avif import register_avif_opener
    register_avif_opener()
    AVIF_SUPPORT = True
except ImportError:
    AVIF_SUPPORT = False
    warnings.warn("AVIF support disabled. Install pillow-avif-plugin for AVIF support.")

# Configure image loading
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, processor, training_args, compute_dtype, device):
    """Modified for Idefics3 architecture with separate processor"""
    # Configure image processor
    if processor is not None and hasattr(processor, 'image_processor'):
        if AVIF_SUPPORT:
            processor.image_processor.image_format = "AVIF"
        else:
            processor.image_processor.image_format = "JPEG"
        processor.image_processor.do_convert_rgb = True

    # Configure vision tower components
    vision_tower = model.model.vision_model
    vision_tower.to(dtype=compute_dtype, device=device)
    
    vision_model_params = vision_tower.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle connector
    connector_params = model.model.connector.parameters()
    set_requires_grad(connector_params, training_args.tune_connector)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.text_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def validate_image_files(data_args):
    """Check image formats and warn about unsupported types"""
    valid_extensions = {'.avif', '.jpg', '.jpeg', '.png', '.webp'}
    invalid_files = []
    
    for img_file in pathlib.Path(data_args.image_folder).rglob('*'):
        if img_file.suffix.lower() not in valid_extensions:
            invalid_files.append(img_file)
    
    if invalid_files:
        warning_msg = f"Found {len(invalid_files)} files with unsupported extensions:\n"
        warning_msg += "\n".join(str(f) for f in invalid_files[:5])
        if len(invalid_files) > 5:
            warning_msg += f"\n...and {len(invalid_files)-5} more"
        
        if data_args.strict_image_validation:
            raise ValueError(f"Invalid image formats detected:\n{warning_msg}")
        else:
            rank0_print(f"WARNING: {warning_msg}")
            rank0_print("Skipping invalid files as strict_image_validation=False")

def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Validate image formats before starting training
    validate_image_files(data_args)

    assert not (training_args.lora_enable and training_args.freeze_llm), 'When using LoRA, the LLM should not be frozen. If you want to freeze the LLM, please disable LoRA.'

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."

    if training_args.lora_namespan_exclude is not None:
        training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
    else:
        training_args.lora_namespan_exclude = []

    if not training_args.vision_lora:
        training_args.lora_namespan_exclude += ["vision_model"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # Initialize processor early for image config
    processor = AutoProcessor.from_pretrained(
        model_args.model_id,
        padding_side="right"
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["vision_model", "connector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        _attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "eager", 
        **bnb_model_from_pretrained_args
    )

    # Configure components with processor reference
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, processor, training_args, compute_dtype, training_args.device)

    model.config.use_cache = False

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    # Configure processor settings
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.vision_lr = training_args.vision_lr

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(
        processor=processor,
        data_args=data_args
    )

    trainer = SmolVLMTrainer(
        model=model,
        args=training_args,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
