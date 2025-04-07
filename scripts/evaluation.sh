#!/bin/bash
# -----------------------------------------------------------------------------
# evaluation.sh
#
# This script serves as a wrapper to run the evaluation process for the 
# translation model. It parses command-line options, validates required inputs,
# and constructs the command to invoke the evaluation.py script with the proper
# parameters.
#
# Usage:
#   ./evaluation.sh [options]
#
# Options:
#   -m, --model_type <base|lora>    Specify the model type to use (default: base).
#   -l, --lora_adapter_dir <path>           Path to the LoRA adapter directory (required if model_type is lora).
#   -i, --input_file <path>         Path to the JSON file containing evaluation data (required).
#   -q, --quantization <4-bit|8-bit> Enable quantization.
#   -a, --attention <flash_attention_2|eager> Attention implementation to use.
#   -h, --help                      Show this help message.
# -----------------------------------------------------------------------------

# Set default parameter values
MODEL_TYPE="base"
LORA_ADAPTER_DIR=""
INPUT_FILE=""
QUANTIZATION=""
ATTENTION="flash_attention_2"

# -----------------------------------------------------------------------------
# Parse command-line arguments using a while-case loop.
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model_type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    -l|--lora_adapter_dir)
      LORA_ADAPTER_DIR="$2"
      shift 2
      ;;
    -i|--input_file)
      INPUT_FILE="$2"
      shift 2
      ;;
    -q|--quantization)
      QUANTIZATION="$2"
      shift 2
      ;;
    -a|--attention)
      ATTENTION="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  -m, --model_type <base|lora>    Specify the model type to use (default: base)."
      echo "  -l, --lora_adapter_dir <path>           Path to the LoRA adapter directory (required if model_type is lora)."
      echo "  -i, --input_file <path>         Path to the JSON file containing evaluation data (required)."
      echo "  -q, --quantization <4-bit|8-bit> Enable quantization."
      echo "  -a, --attention <flash_attention_2|eager> Attention implementation to use."
      echo "  -h, --help                      Show this help message."
      exit 0
      ;;
    *)
      echo "Error: Unknown option '$1'"
      echo "Use '$0 --help' for usage information."
      exit 1
      ;;
  esac
done

# -----------------------------------------------------------------------------
# Validate required options
# -----------------------------------------------------------------------------
if [ -z "$INPUT_FILE" ]; then
  echo "Error: --input_file is required."
  echo "Use '$0 --help' for usage information."
  exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Input file not found at '$INPUT_FILE'."
  exit 1
fi

if [ "$MODEL_TYPE" == "lora" ] && [ -z "$LORA_ADAPTER_DIR" ]; then
  echo "Error: --lora_adapter_dir must be specified when --model_type is set to 'lora'."
  echo "Use '$0 --help' for usage information."
  exit 1
fi

# -----------------------------------------------------------------------------
# Construct the command to run the evaluation.py script with the provided options.
# -----------------------------------------------------------------------------
python_command="python evaluation.py --model_type $MODEL_TYPE --input_file \"$INPUT_FILE\""

# Append LoRA adapter directory if provided
if [ -n "$LORA_ADAPTER_DIR" ]; then
  python_command="$python_command --lora_adapter_dir \"$LORA_ADAPTER_DIR\""
fi

# Append quantization flag if specified
if [ -n "$QUANTIZATION" ]; then
  python_command="$python_command --quantization $QUANTIZATION"
fi

# Append attention implementation flag if specified (default is set)
if [ -n "$ATTENTION" ]; then
  python_command="$python_command --attn_implementation $ATTENTION"
fi

# -----------------------------------------------------------------------------
# Display and execute the command.
# -----------------------------------------------------------------------------
echo "Running: $python_command"
eval "$python_command"

# Exit with the status code returned by the Python script
exit $?
