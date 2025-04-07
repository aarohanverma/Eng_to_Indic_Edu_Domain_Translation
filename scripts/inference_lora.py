import argparse
import json
import os
import gc
from datetime import datetime
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor

# Sentence tokenization utilities
from nltk import sent_tokenize
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA
from mosestokenizer import MosesSentenceSplitter

# Global constants
BASE_MODEL_CKPT_DIR = "ai4bharat/indictrans2-en-indic-1B"
INDIC_INDIC_CKPT_DIR = "ai4bharat/indictrans2-indic-indic-1B"
BATCH_SIZE = 16  # Batch size for inference (can be adjusted)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Supported target languages and language code mappings
ALLOWED_TARGET_LANGUAGES = ["hindi", "marathi", "gujarati", "tamil"]
TARGET_LANGUAGE_CODES = {
    "hindi": "hin_Deva",
    "marathi": "mar_Deva",
    "gujarati": "guj_Gujr",
    "tamil": "tam_Taml",
}
FLORES_CODES = {
    "asm_Beng": "as",
    "awa_Deva": "hi",
    "ben_Beng": "bn",
    "bho_Deva": "hi",
    "brx_Deva": "hi",
    "doi_Deva": "hi",
    "eng_Latn": "en",
    "gom_Deva": "kK",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "hne_Deva": "hi",
    "kan_Knda": "kn",
    "kas_Arab": "ur",
    "kas_Deva": "hi",
    "kha_Latn": "en",
    "lus_Latn": "en",
    "mag_Deva": "hi",
    "mai_Deva": "hi",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "mni_Beng": "bn",
    "mni_Mtei": "hi",
    "npi_Deva": "ne",
    "ory_Orya": "or",
    "pan_Guru": "pa",
    "san_Deva": "hi",
    "sat_Olck": "or",
    "snd_Arab": "ur",
    "snd_Deva": "hi",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
}

# Initialize the IndicProcessor in inference mode
INDIC_PROCESSOR = IndicProcessor(inference=True)


def clear_cache():
    """
    Clears the CUDA cache and collects garbage to free GPU memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"Failed to clear CUDA cache: {e}")


def split_sentences(input_text, lang_code):
    """
    Splits the input text into sentences based on the language code.

    For English (eng_Latn), uses both NLTK and Moses sentence splitters and
    selects the one yielding fewer sentences. For other languages, uses the
    Indic NLP sentence splitting.

    Args:
        input_text (str): The text to be split into sentences.
        lang_code (str): The language code (e.g., "eng_Latn").

    Returns:
        list: A list of sentence strings.
    """
    if lang_code == "eng_Latn":
        # Use both Moses and NLTK splitters for English and choose the shorter output
        sents_moses = MosesSentenceSplitter(FLORES_CODES[lang_code])([input_text])
        sents_nltk = sent_tokenize(input_text)
        input_sentences = sents_nltk if len(sents_nltk) < len(sents_moses) else sents_moses
        # Remove soft hyphen artifacts
        input_sentences = [sent.replace("\xad", "") for sent in input_sentences]
    else:
        # For other languages, use Indic NLP sentence splitting
        input_sentences = sentence_split(
            input_text, lang=FLORES_CODES[lang_code], delim_pat=DELIM_PAT_NO_DANDA
        )
    return input_sentences


def initialize_model_and_tokenizer(base_ckpt_dir, lora_adapter_dir):
    """
    Loads the base transformer model and tokenizer, and applies the LoRA adapter.

    Args:
        base_ckpt_dir (str): Path or identifier for the base model checkpoint.
        lora_adapter_dir (str): Path to the directory containing LoRA adapter weights.

    Returns:
        tuple: (tokenizer, model) if successful; otherwise, (None, None).
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_ckpt_dir, trust_remote_code=True)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_ckpt_dir, trust_remote_code=True)
        # Load the LoRA adapter onto the base model
        model = PeftModel.from_pretrained(base_model, lora_adapter_dir)
        model.eval()
        model.to(DEVICE)
        return tokenizer, model
    except Exception as e:
        print(f"Error initializing model and tokenizer: {e}")
        return None, None


def batch_translate(input_sentences, src_lang_code, tgt_lang_code, model, tokenizer):
    """
    Translates a list of input sentences from the source to target language in batches.

    Args:
        input_sentences (list): List of sentences to translate.
        src_lang_code (str): Source language code (e.g., "eng_Latn").
        tgt_lang_code (str): Target language code (e.g., "hin_Deva").
        model (PeftModel): The LoRA-adapted model.
        tokenizer (AutoTokenizer): The tokenizer for the model.

    Returns:
        list: Translated sentences.
    """
    translations = []
    # Process sentences in batches
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        # Preprocess batch with IndicProcessor
        processed_batch = INDIC_PROCESSOR.preprocess_batch(batch, src_lang=src_lang_code, tgt_lang=tgt_lang_code)
        # Tokenize the processed batch
        inputs = tokenizer(
            processed_batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)
        # Generate translations using beam search
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=10,
                num_return_sequences=1,
                repetition_penalty=1.2, 
                early_stopping=True,
            )
        # Decode generated tokens
        with tokenizer.as_target_tokenizer():
            decoded_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        # Postprocess translations
        translations += INDIC_PROCESSOR.postprocess_batch(decoded_tokens, lang=tgt_lang_code)
        del inputs  # Free GPU memory
        clear_cache()  # Clear CUDA cache to avoid memory issues
    return translations


def translate_text(input_text, source_language, target_language, lora_adapter_dir):
    """
    Translates input text from English to the specified target language using a LoRA adapter.

    Args:
        input_text (str): The English text to translate.
        source_language (str): Should be "english" (case-insensitive).
        target_language (str): Target language (e.g., "hindi", "marathi").
        lora_adapter_dir (str): Path to the LoRA adapter weights.

    Returns:
        str or None: The translated text, or None if an error occurs.
    """
    # Validate source language
    if source_language.lower() != "english":
        print(f"Error: Translation from '{source_language}' is not supported. Only English is allowed.")
        return None

    # Validate target language
    if target_language.lower() not in ALLOWED_TARGET_LANGUAGES:
        allowed = ", ".join(ALLOWED_TARGET_LANGUAGES)
        print(f"Error: Translation to '{target_language}' is not supported. Allowed target languages are: {allowed}.")
        return None

    src_lang_code = "eng_Latn"
    tgt_lang_code = TARGET_LANGUAGE_CODES[target_language.lower()]

    # Choose the appropriate base model checkpoint
    base_model_ckpt = BASE_MODEL_CKPT_DIR
    if tgt_lang_code != "hin_Deva":
        base_model_ckpt = INDIC_INDIC_CKPT_DIR

    # Initialize model and tokenizer with LoRA adapter
    tokenizer, model = initialize_model_and_tokenizer(base_model_ckpt, lora_adapter_dir)
    if model is None or tokenizer is None:
        return None

    # Split input text into sentences and translate in batches
    input_sentences = split_sentences(input_text, src_lang_code)
    translated_sentences = batch_translate(input_sentences, src_lang_code, tgt_lang_code, model, tokenizer)
    return " ".join(translated_sentences)


def translate_to_all(input_texts, source_language, lora_adapter_dir):
    """
    Translates a list of texts to all supported target languages using a LoRA adapter.

    Args:
        input_texts (list): List of texts to be translated.
        source_language (str): Must be "english" (case-insensitive).
        lora_adapter_dir (str): Path to the LoRA adapter weights.

    Returns:
        dict: Dictionary mapping each target language to its list of translations.
    """
    if source_language.lower() != "english":
        print(f"Error: Translation from '{source_language}' is not supported. Only English is allowed.")
        return {}

    all_translations = {}
    for target_language in ALLOWED_TARGET_LANGUAGES:
        translations_for_lang = []
        for input_text in input_texts:
            translated_text = translate_text(input_text, source_language, target_language, lora_adapter_dir)
            translations_for_lang.append(translated_text if translated_text is not None else None)
        all_translations[target_language] = translations_for_lang
    return all_translations


def process_input(input_string, input_file):
    """
    Processes input from a direct string or a JSON file.

    Args:
        input_string (str): Input text provided via command line.
        input_file (str): Path to a JSON file containing input data.

    Returns:
        list or None: A list of input texts, or None if processing fails.
    """
    if input_string:
        return [input_string]
    elif input_file:
        if os.path.exists(input_file):
            print(f"Processing input from file: {input_file}")
            with open(input_file, 'r', encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, str):
                        return [data]
                    else:
                        print("Error: JSON file should contain a list of strings or a single string.")
                        return None
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from '{input_file}'.")
                    return None
        else:
            print(f"Error: Input file not found at '{input_file}'.")
            return None
    else:
        print("Error: No input provided. Please use --input_string or --input_file.")
        return None


def save_translations(translations, input_texts, output_file):
    """
    Saves translation results to a file with a timestamp and structured output.

    Args:
        translations (dict): Dictionary mapping target languages to translation lists.
        input_texts (list): List of original input texts.
        output_file (str): Path to the output file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Translation Output ({timestamp})\n\n")
        for target_lang, translated_texts in translations.items():
            f.write(f"## Target Language: {target_lang.capitalize()}\n")
            for i, translated_text in enumerate(translated_texts):
                f.write(f"### Input Sentence {i+1}\n")
                f.write(f"```\n{input_texts[i]}\n```\n\n")
                f.write(f"### Translated Sentence {i+1}\n")
                f.write(f"```\n{translated_text}\n```\n\n")
        f.write("# End of Translation Output\n")
    print(f"Translations saved to: {output_file}")


def main():
    """
    Main function to handle command-line arguments and execute the translation process.
    """
    parser = argparse.ArgumentParser(
        description="Translate English text to Hindi, Marathi, Gujarati, or Tamil using LoRA adapters."
    )
    parser.add_argument("--source_language", type=str, default="english", help="Source language (default: english)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target_language", type=str, help="Target language (hindi, marathi, gujarati, tamil)")
    group.add_argument("--translate_to_all_targets", action="store_true", help="Translate to all supported target languages")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_string", type=str, help="Input string to translate")
    input_group.add_argument("--input_file", type=str, help="Path to a JSON file containing input (list or single string)")
    parser.add_argument("--output_file", type=str, help="Path to save the output file (optional)")
    parser.add_argument("--lora_adapter_dir", type=str, required=True, help="Path to the LoRA adapter directory")
    args = parser.parse_args()

    # Validate the LoRA adapter directory
    if not os.path.isdir(args.lora_adapter_dir):
        print(f"Error: LoRA adapter directory not found at '{args.lora_adapter_dir}'.")
        return

    input_texts = process_input(args.input_string, args.input_file)
    if not input_texts:
        return

    if args.translate_to_all_targets:
        translations = translate_to_all(input_texts, args.source_language, args.lora_adapter_dir)
        if translations:
            print("\n--- Translations to All Target Languages ---")
            for target_lang, translated_texts in translations.items():
                print(f"\nTarget Language: {target_lang.capitalize()}")
                for i, translated_text in enumerate(translated_texts):
                    print(f"\nInput Sentence {i+1}:")
                    print(f"```\n{input_texts[i]}\n```")
                    print(f"Translated Sentence {i+1}:")
                    print(f"```\n{translated_text}\n```")
            if args.output_file:
                save_translations(translations, input_texts, args.output_file)
    elif args.target_language:
        target_language = args.target_language.lower()
        translations = {target_language: []}
        for i, input_text in enumerate(input_texts):
            translated_text = translate_text(input_text, args.source_language, target_language, args.lora_adapter_dir)
            if translated_text:
                translations[target_language].append(translated_text)
                print(f"\n--- Translation to {target_language.capitalize()} ---")
                print(f"Input Sentence {i+1}:")
                print(f"```\n{input_text}\n```")
                print(f"Translated Sentence {i+1}:")
                print(f"```\n{translated_text}\n```")
            else:
                translations[target_language].append(None)
        if args.output_file:
            save_translations(translations, input_texts, args.output_file)


if __name__ == "__main__":
    main()
