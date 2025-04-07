import argparse
import json
import os
from datetime import datetime
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
import evaluate

# Import the inference module which contains functions for model initialization and translation
import inference

# Global constants
BATCH_SIZE = 4  # Number of sentences processed in a batch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Allowed target languages and corresponding language codes are imported from inference
ALLOWED_TARGET_LANGUAGES = inference.ALLOWED_TARGET_LANGUAGES
TARGET_LANGUAGE_CODES = inference.TARGET_LANGUAGE_CODES


def load_evaluation_data(input_file: str) -> list | None:
    """
    Loads evaluation data from a JSON file.
    
    The input file is expected to be a JSON array where each element is a dictionary
    containing an 'english' key and one or more target language keys.
    
    Args:
        input_file (str): Path to the evaluation JSON file.
        
    Returns:
        list | None: List of evaluation items if successful; otherwise, None.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'.")
        return None
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Validate that the data is a list of dictionaries containing the 'english' key.
            if not isinstance(data, list):
                print("Error: Input file should contain a list of JSON objects.")
                return None
            for item in data:
                if not isinstance(item, dict) or 'english' not in item:
                    print("Error: Each JSON object should contain at least an 'english' key.")
                    return None
            return data
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file}'.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        return None


def main():
    """
    Main function to parse command-line arguments, load evaluation data,
    perform translations using the inference module, and evaluate the translations.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate English to Indic translation using a model and reference translations."
    )
    parser.add_argument("--model_type", type=str, default="base", choices=["base", "lora"],
                        help="Specify the model type to use ('base' or 'lora'). Default: base.")
    parser.add_argument("--lora_adapter_dir", type=str,
                        help="Path to the LoRA adapter directory (required if --model_type is 'lora').")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the JSON file containing English sentences and reference translations.")
    parser.add_argument("--quantization", type=str, choices=["4-bit", "8-bit", ""], default="",
                        help="Enable quantization ('4-bit' or '8-bit').")
    parser.add_argument("--attn_implementation", type=str, choices=["flash_attention_2", "eager"],
                        default="flash_attention_2",
                        help="Attention implementation to use ('flash_attention_2' or 'eager').")

    args = parser.parse_args()

    # Validate LoRA argument when using LoRA model type
    if args.model_type == "lora" and args.lora_adapter_dir is None:
        parser.error("--lora_adapter_dir must be specified when --model_type is set to 'lora'.")

    # Load evaluation data from JSON file
    evaluation_data = load_evaluation_data(args.input_file)
    if not evaluation_data:
        return

    # Identify target languages to evaluate based on keys in the first data entry
    target_languages_to_evaluate = [lang for lang in ALLOWED_TARGET_LANGUAGES if lang in evaluation_data[0]]
    if not target_languages_to_evaluate:
        print("Warning: No target languages found in the input file for evaluation.")
        return

    print(f"Evaluating for the following target languages found in the input file: "
          f"{', '.join(target_languages_to_evaluate)}")

    # Initialize the two translation models (English-to-Indic and Indic-to-Indic)
    # For Hindi, we use the English-to-Indic model; for other languages, we use the Indic-to-Indic model.
    en_indic_tokenizer, en_indic_model = inference.initialize_model_and_tokenizer(
        inference.EN_INDIC_CKPT_DIR,
        quantization=args.quantization,
        attn_implementation=args.attn_implementation,
        lora_adapter_dir=args.lora_adapter_dir if args.model_type == "lora" and "hindi" in target_languages_to_evaluate else None
    )
    indic_indic_tokenizer, indic_indic_model = inference.initialize_model_and_tokenizer(
        inference.INDIC_INDIC_CKPT_DIR,
        quantization=args.quantization,
        attn_implementation=args.attn_implementation,
        lora_adapter_dir=args.lora_adapter_dir if args.model_type == "lora" and any(lang != "hindi" for lang in target_languages_to_evaluate) else None
    )

    if not all([en_indic_model, en_indic_tokenizer, indic_indic_model, indic_indic_tokenizer]):
        print("Error initializing models and tokenizers.")
        return

    print(f"Evaluating using {args.model_type} model...")

    # Load the CHRF metric for evaluation using the evaluate library
    chrf_metric = evaluate.load("chrf")
    overall_results = {}  # Dictionary to hold predictions and references per target language

    # Iterate through each evaluation data item
    for item in evaluation_data:
        english_sentence = item.get('english')
        if not english_sentence:
            print("Warning: Skipping item with missing 'english' sentence.")
            continue

        # Loop through each target language present in the evaluation data
        for target_lang in target_languages_to_evaluate:
            reference_translation = item.get(target_lang)
            if not reference_translation:
                print(f"Warning: Skipping evaluation for '{target_lang}' as reference translation is missing for sentence: '{english_sentence}'.")
                continue

            # Select the appropriate model and tokenizer based on the target language
            model_to_use = en_indic_model if target_lang.lower() == "hindi" else indic_indic_model
            tokenizer_to_use = en_indic_tokenizer if target_lang.lower() == "hindi" else indic_indic_tokenizer

            # Translate the English sentence using the inference.translate_text function
            predicted_translation = inference.translate_text(
                english_sentence, "english", target_lang, model_to_use, tokenizer_to_use
            )

            if predicted_translation is not None:
                # Append predictions and references for evaluation
                overall_results.setdefault(target_lang, {"predictions": [], "references": []})
                overall_results[target_lang]["predictions"].append(predicted_translation)
                overall_results[target_lang]["references"].append(reference_translation)
            else:
                print(f"Error: Translation failed for sentence '{english_sentence}' to '{target_lang}'.")

    # Print overall evaluation results using the CHRF metric
    print("\n--- Overall Evaluation Results (chrF) ---")
    for lang, data in overall_results.items():
        if data["predictions"] and data["references"]:
            chrf_results = chrf_metric.compute(predictions=data["predictions"], references=data["references"])
            print(f"Language: {lang.capitalize()}")
            print(f"  chrF score: {chrf_results['score']:.4f}")
        else:
            print(f"Warning: No predictions or references found for {lang.capitalize()}. Skipping evaluation.")


if __name__ == "__main__":
    main()
