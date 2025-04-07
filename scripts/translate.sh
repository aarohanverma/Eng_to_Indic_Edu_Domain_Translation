#!/bin/bash
# -----------------------------------------------------------------------------
# translate.sh
#
# This script serves as a wrapper for running translation inference. It allows
# the user to choose between using a base model or a LoRA-adapted model, to translate
# either to a specific target language or to all supported languages.
#
# Usage:
#   ./translate.sh [options]
#
# Options:
#   -m <base|lora>           Specify model type (default: base).
#   --lora_adapter_dir <path> Path to LoRA adapter directory (required if -m lora).
#   -t <lang>                Translate to a specific target language (hindi, marathi, gujarati, tamil).
#   -a                       Translate to all supported languages.
#   -s <text>                Input text to translate.
#   -f <file>                Path to an input JSON file (expects a 'text' key or list).
#   -o <file>                Output file name.
#   -h                       Show this help message.
#
# Examples:
#   bash translate.sh -t hindi -s "Hello." 
#   bash translate.sh -m lora --lora_adapter_dir "/path/to/adapter" -t marathi -s "Hi."
#   bash translate.sh -a -s "Translate this."
#   bash translate.sh -m lora --lora_adapter_dir "/path/to/adapter" -t gujarati -f datasets/test.json -o output.txt
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Define project directories based on script location.
# -----------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
LORA_ADAPTER_DIR="$PROJECT_ROOT/lora_adapters"

# -----------------------------------------------------------------------------
# Set default output file and flag to detect user-specified output.
# -----------------------------------------------------------------------------
DEFAULT_OUTPUT_FILE="translations.txt"
output_file_specified=false

# -----------------------------------------------------------------------------
# Function: show_help
# Displays help and usage information.
# -----------------------------------------------------------------------------
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m <base|lora>           Specify model type (default: base)."
    echo "  --lora_adapter_dir <path>  Path to LoRA adapter directory (required if -m lora)."
    echo "  -t <lang>                Translate to: hindi, marathi, gujarati, tamil."
    echo "  -a                       Translate to all supported languages."
    echo "  -s <text>                Input text to translate."
    echo "  -f <file>                Path to input JSON file (expects a 'text' key or list)."
    echo "  -o <file>                Output file name."
    echo "  -h                       Show this help message."
    echo ""
    echo "Examples:"
    echo "  Translate to Hindi (base model): bash $0 -t hindi -s 'Hello.'"
    echo "  Translate to Marathi (LoRA model): bash $0 -m lora --lora_adapter_dir '$LORA_ADAPTER_DIR/adapter_checkpoint' -t marathi -s 'Hi.'"
    echo "  Translate to all (base model): bash $0 -a -s 'Translate this.'"
    echo "  Translate from JSON (LoRA model): bash $0 -m lora --lora_adapter_dir '$LORA_ADAPTER_DIR/my_adapter' -t gujarati -f datasets/testing_dataset/test_data.json -o output.txt"
}

# -----------------------------------------------------------------------------
# Initialize option variables with defaults.
# -----------------------------------------------------------------------------
model_type="base"
lora_adapter_dir=""
target_language=""
translate_all=false
input_string=""
input_file=""
output_file="$DEFAULT_OUTPUT_FILE"

# -----------------------------------------------------------------------------
# Parse command-line options.
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m)
            model_type="$2"
            shift 2
            ;;
        --lora_adapter_dir)
            lora_adapter_dir="$2"
            shift 2
            ;;
        -t)
            target_language="$2"
            shift 2
            ;;
        -a)
            translate_all=true
            shift
            ;;
        -s)
            input_string="$2"
            shift 2
            ;;
        -f)
            input_file="$2"
            shift 2
            ;;
        -o)
            output_file="$2"
            output_file_specified=true
            shift 2
            ;;
        -h)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Basic validation of inputs.
# -----------------------------------------------------------------------------
if [[ "$model_type" != "base" && "$model_type" != "lora" ]]; then
    echo "Error: Model type must be 'base' or 'lora'."
    show_help
    exit 1
fi

if [[ "$model_type" == "lora" && -z "$lora_adapter_dir" ]]; then
    echo "Error: For 'lora' model, please specify --lora_adapter_dir."
    show_help
    exit 1
fi

if [[ -n "$target_language" && "$translate_all" == "true" ]]; then
    echo "Error: Cannot use both -t and -a options simultaneously."
    show_help
    exit 1
fi

if [[ -z "$input_string" && -z "$input_file" ]]; then
    echo "Error: Please provide input using -s (string) or -f (file)."
    show_help
    exit 1
fi

if [[ -n "$input_string" && -n "$input_file" ]]; then
    echo "Error: Cannot use both -s and -f options at the same time."
    show_help
    exit 1
fi

# -----------------------------------------------------------------------------
# Determine which Python inference script to use based on model type.
# -----------------------------------------------------------------------------
python_script="$SCRIPTS_DIR/inference.py"
if [[ "$model_type" == "lora" ]]; then
    python_script="$SCRIPTS_DIR/inference_lora.py"
fi

# -----------------------------------------------------------------------------
# Construct the Python command with all specified options.
# -----------------------------------------------------------------------------
python_command="python $python_script --source_language 'english'"

if [[ "$translate_all" == "true" ]]; then
    python_command="$python_command --translate_to_all_targets"
elif [[ -n "$target_language" ]]; then
    python_command="$python_command --target_language '$target_language'"
fi

if [[ -n "$input_string" ]]; then
    python_command="$python_command --input_string '$input_string'"
elif [[ -n "$input_file" ]]; then
    # If the input file path is relative, prepend the project root.
    if [[ ! "$input_file" =~ ^/ ]]; then
        input_file="$PROJECT_ROOT/$input_file"
    fi
    python_command="$python_command --input_file '$input_file'"
fi

# Add the output file argument if specified
if [[ "$output_file_specified" == "true" ]]; then
    python_command="$python_command --output_file '$output_file'"
fi

# Add the LoRA adapter directory if using a LoRA model
if [[ "$model_type" == "lora" ]]; then
    python_command="$python_command --lora_adapter_dir '$lora_adapter_dir'"
fi

# -----------------------------------------------------------------------------
# Execute the constructed Python command.
# -----------------------------------------------------------------------------
echo "Running: $python_command"
eval "$python_command"

exit 0
