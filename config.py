import os
import argparse
from os.path import join as ospj
from util_data import SUBSET_NAMES

def int2none(v):
    if v is None or v == "":
        return None
    elif v.lower() in ("none", "null"):
        return None
    else:
        return int(v)

def str2none(v):
    if v is None or v == "":
        return v
    elif v.lower() in ("none", "null"):
        return None
    else:
        return str(v)

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    elif v is None or v == "":
        return None
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training script with natural language captions.")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="eurosat")
    parser.add_argument("--n_shot", type=int2none, default=16)
    parser.add_argument("--fewshot_seed", type=str, default="seed0")
    parser.add_argument("--target_class_idx", type=int2none, default=None)
    parser.add_argument("--caption_path", type=str, default=None, 
                       help="Path to the captions JSON file")
    parser.add_argument("--fewshot_data_dir", type=str, required=True,
                       help="Directory containing the few-shot training data")
    
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument("--output_dir", type=str2none, default="outputs")
    parser.add_argument("--resume_from_checkpoint", type=str2none, default=None)
    
    # Optimizer parameters
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--scale_lr", action="store_true")
    
    # Model parameters
    parser.add_argument("--train_text_encoder", type=str2bool, default=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                       help="Path to the pretrained model")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    
    # Training infrastructure
    parser.add_argument("--is_tqdm", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--allow_tf32", action="store_true")
    
    # Advanced training options
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--rank", type=int, default=16)
    
    # Validation parameters
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_epochs", type=int, default=50)
    parser.add_argument("--validation_images", nargs="+", type=str, default=None)
    
    # Data processing
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    
    # Checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    
    # HuggingFace Hub integration
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    
    # Tokenizer
    parser.add_argument("--tokenizer_max_length", type=int, default=None)
    parser.add_argument("--text_encoder_use_attention_mask", action="store_true")
    
    # Adam Optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    
    args = parser.parse_args(input_args) if input_args is not None else parser.parse_args()
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
        
    # Set output directory
    mid1 = f"shot{args.n_shot}_{args.fewshot_seed}"
    if not args.train_text_encoder:
        mid1 += "_notextlora"
    mid2 = "dataset-wise" if args.target_class_idx is None else SUBSET_NAMES[args.dataset][args.target_class_idx]
    
    num_run = f"_epoch{args.num_train_epochs}" if args.num_train_epochs is not None else ""
    ddrank = "" if args.rank == 16 else f"_ddrank{args.rank}"
    
    args.output_dir = ospj(
        args.output_dir,
        args.dataset,
        mid1,
        f"lr{args.learning_rate}{num_run}{ddrank}",
        mid2,
    )
    
    return args