#!/bin/bash
# -----------------------------------------------------------------------------
# train.sh
#
# This script is a simplified launcher for training LoRA adapters.
# It sets default parameters (data directory, model name, output directory,
# and language lists) and builds a command to execute the Python training script.
#
# Usage: ./train.sh [options]
#   -d <dir>            Data directory (default set in the script)
#   -m <model_name>     Model name (default set in the script)
#   -o <dir>            Output directory (default set in the script)
#   --src <langs>       Comma-separated list of source languages
#   --tgt <langs>       Comma-separated list of target languages
#   --print_samples     Flag to print sample predictions during evaluation
#   -h, --help          Show this help message and exit
# -----------------------------------------------------------------------------

# Resolve the directory paths for the script and project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Path to the training Python script.
training_script="$PROJECT_ROOT/scripts/train_lora.py"

# -----------------------------------------------------------------------------
# Default configuration values.
# -----------------------------------------------------------------------------
DEFAULT_DATA_DIR="$PROJECT_ROOT/datasets/train_validation/en-indic-exp"
DEFAULT_MODEL_NAME="ai4bharat/indictrans2-en-indic-1B"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/lora_adapters/output_$(date +%Y%m%d_%H%M%S)"
DEFAULT_SRC_LANG_LIST="eng_Latn"
DEFAULT_TGT_LANG_LIST="guj_Gujr,hin_Deva,mar_Deva,tam_Taml"

# -----------------------------------------------------------------------------
# Function: show_help
# Displays usage information for this script.
# -----------------------------------------------------------------------------
show_help() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -d <dir>            Data directory (default: $DEFAULT_DATA_DIR)"
  echo "  -m <model_name>     Model name (default: $DEFAULT_MODEL_NAME)"
  echo "  -o <dir>            Output directory (default: $DEFAULT_OUTPUT_DIR)"
  echo "  --src <langs>       Source language list (default: $DEFAULT_SRC_LANG_LIST)"
  echo "  --tgt <langs>       Target language list (default: $DEFAULT_TGT_LANG_LIST)"
  echo "  --print_samples     Print samples during evaluation"
  echo "  -h, --help          Show this help message"
  echo ""
}

# -----------------------------------------------------------------------------
# Parse command-line arguments.
# -----------------------------------------------------------------------------
data_dir="$DEFAULT_DATA_DIR"
model_name="$DEFAULT_MODEL_NAME"
output_dir="$DEFAULT_OUTPUT_DIR"
src_lang_list="$DEFAULT_SRC_LANG_LIST"
tgt_lang_list="$DEFAULT_TGT_LANG_LIST"
print_samples=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d) data_dir="$2"; shift 2 ;;
    -m) model_name="$2"; shift 2 ;;
    -o) output_dir="$2"; shift 2 ;;
    --src) src_lang_list="$2"; shift 2 ;;
    --tgt) tgt_lang_list="$2"; shift 2 ;;
    --print_samples) print_samples=true; shift ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown option: $1"; show_help; exit 1 ;;
  esac
done

# -----------------------------------------------------------------------------
# Verify that the training Python script exists.
# -----------------------------------------------------------------------------
if [[ ! -f "$training_script" ]]; then
  echo "Error: Training script not found at $training_script."
  exit 1
fi

# Create the output directory if it does not exist.
mkdir -p "$output_dir" || { echo "Failed to create output directory: $output_dir."; exit 1; }

# -----------------------------------------------------------------------------
# Define default training arguments as an array (flag/value pairs).
# -----------------------------------------------------------------------------
ARGS=(
  --save_steps 500 --eval_steps 500 --batch_size 32
  --num_train_epochs 10 --max_steps 5000 --grad_accum_steps 2
  --warmup_steps 200 --max_grad_norm 1.0 --learning_rate 3e-4
  --weight_decay 0.01 --adam_beta1 0.9 --adam_beta2 0.98
  --dropout 0.1 --optimizer adamw_torch --lr_scheduler inverse_sqrt
  --label_smoothing 0.05 --num_workers 4
  --metric_for_best_model eval_chrF --greater_is_better
  --lora_target_modules "q_proj,k_proj,v_proj,out_proj,fc1,fc2"
  --lora_dropout 0.05 --lora_r 8 --lora_alpha 16
  --patience 4 --threshold 1.0
  --report_to none
)

# -----------------------------------------------------------------------------
# Build the command string for running the training script.
# -----------------------------------------------------------------------------
cmd="python3 \"$training_script\""
cmd+=" --data_dir \"$data_dir\""
cmd+=" --model \"$model_name\""
cmd+=" --output_dir \"$output_dir\""
cmd+=" --src_lang_list \"$src_lang_list\""
cmd+=" --tgt_lang_list \"$tgt_lang_list\""

# Append additional arguments from the ARGS array.
for ((i=0; i<${#ARGS[@]}; i+=2)); do
  cmd+=" ${ARGS[i]} ${ARGS[i+1]}"
done

# Append the --print_samples flag if requested.
if $print_samples; then
  cmd+=" --print_samples"
fi

# -----------------------------------------------------------------------------
# Execute the training command.
# -----------------------------------------------------------------------------
echo "Running training command:"
echo "$cmd"
eval "$cmd" || { echo "Training failed."; exit 1; }

echo "LoRA training completed successfully. Output directory: $output_dir"
exit 0
