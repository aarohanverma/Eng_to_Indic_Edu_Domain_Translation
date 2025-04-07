import argparse
import json
import os
import gc
from datetime import datetime
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor
from nltk import sent_tokenize
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA
from mosestokenizer import MosesSentenceSplitter

# Global constants for model checkpoints and device
EN_INDIC_CKPT_DIR = "ai4bharat/indictrans2-en-indic-1B"
INDIC_INDIC_CKPT_DIR = "ai4bharat/indictrans2-indic-indic-1B"
BATCH_SIZE = 4  # Batch size for translation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Supported target languages and their language codes
ALLOWED_TARGET_LANGUAGES = ["hindi", "marathi", "gujarati", "tamil"]
TARGET_LANGUAGE_CODES = {
    "hindi": "hin_Deva",
    "marathi": "mar_Deva",
    "gujarati": "guj_Gujr",
    "tamil": "tam_Taml",
}

# FLORES language codes mapping (used for sentence splitting)
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

# Initialize IndicProcessor for preprocessing and postprocessing
INDIC_PROCESSOR = IndicProcessor(inference=True)


def clear_cache() -> None:
    """
    Clears the Python garbage collector and CUDA cache (if available).
    
    This helps in managing GPU memory by releasing unused resources.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"Failed to clear CUDA cache: {e}")


def split_sentences(input_text: str, lang_code: str) -> list:
    """
    Splits the input text into sentences based on the source language code.
    
    For English, both NLTK and Moses sentence splitters are used and the one yielding fewer sentences is chosen.
    For other languages, indicnlp's sentence_split is used.
    
    Args:
        input_text (str): The text to split.
        lang_code (str): The language code of the input text (e.g., "eng_Latn").
        
    Returns:
        list: A list of sentence strings.
    """
    if lang_code == "eng_Latn":
        # Use MosesSentenceSplitter and NLTK's sent_tokenize for English
        with MosesSentenceSplitter(FLORES_CODES[lang_code]) as splitter:
            sents_moses = splitter([input_text])
        sents_nltk = sent_tokenize(input_text)
        # Choose the sentence split that results in fewer splits (assumed to be more accurate)
        input_sentences = sents_nltk if len(sents_nltk) < len(sents_moses) else sents_moses
        # Remove any soft hyphen artifacts from the sentences
        input_sentences = [sent.replace("\xad", "") for sent in input_sentences]
    else:
        # For other languages, use the indicnlp sentence splitter with a custom delimiter pattern
        input_sentences = sentence_split(
            input_text, lang=FLORES_CODES[lang_code], delim_pat=DELIM_PAT_NO_DANDA
        )
    return input_sentences


def _get_quantization_config(quantization: str) -> BitsAndBytesConfig | None:
    """
    Returns the quantization configuration based on the given quantization type.
    
    Args:
        quantization (str): Quantization type ("4-bit", "8-bit", or empty string).
        
    Returns:
        BitsAndBytesConfig or None: The configuration for quantization or None if not specified.
    """
    if quantization == "4-bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    return None


def initialize_model_and_tokenizer(ckpt_dir: str, quantization: str = "", attn_implementation: str = "flash_attention_2",
                                   lora_adapter_dir: str | None = None) -> tuple:
    """
    Initializes and returns the tokenizer and model from the given checkpoint directory.
    
    This function applies optional quantization and LoRA adapter weights if specified.
    It also sets the attention implementation based on availability.
    
    Args:
        ckpt_dir (str): Checkpoint directory for the pretrained model.
        quantization (str, optional): Type of quantization to apply ("4-bit" or "8-bit"). Defaults to "".
        attn_implementation (str, optional): Attention implementation ("flash_attention_2" or "eager"). Defaults to "flash_attention_2".
        lora_adapter_dir (str | None, optional): Directory for LoRA adapter weights. Defaults to None.
        
    Returns:
        tuple: A tuple containing the tokenizer and model.
    """
    # Obtain quantization configuration (if any)
    qconfig = _get_quantization_config(quantization)

    # Check if flash attention 2 is available; if not, fallback to 'eager'
    if attn_implementation == "flash_attention_2":
        if not (is_flash_attn_2_available() and is_flash_attn_greater_or_equal_2_10()):
            attn_implementation = "eager"

    # Load tokenizer from pretrained checkpoint with trust_remote_code enabled
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    # Load the model with specified attention implementation, low CPU memory usage, and quantization config
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    # If a LoRA adapter directory is provided, apply the adapter weights
    if lora_adapter_dir:
        model = PeftModel.from_pretrained(model, lora_adapter_dir)

    # Move the model to the appropriate device and convert to half precision if not quantized
    if qconfig is None:
        model = model.to(DEVICE).half()
    else:
        model = model.to(DEVICE)

    model.eval()  # Set the model to evaluation mode
    return tokenizer, model


def batch_translate(input_sentences: list, src_lang_code: str, tgt_lang_code: str, model, tokenizer) -> list:
    """
    Translates a list of sentences from the source language to the target language in batches.
    
    This function preprocesses the input sentences, tokenizes them, performs translation using beam search,
    decodes the generated tokens, and then postprocesses the translations.
    
    Args:
        input_sentences (list): List of sentences to translate.
        src_lang_code (str): Source language code (e.g., "eng_Latn").
        tgt_lang_code (str): Target language code (e.g., "hin_Deva").
        model: The translation model.
        tokenizer: The tokenizer associated with the model.
        
    Returns:
        list: A list of translated sentence strings.
    """
    translations = []
    # Process sentences in batches
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i: i + BATCH_SIZE]
        # Preprocess the batch using IndicProcessor
        batch = INDIC_PROCESSOR.preprocess_batch(batch, src_lang=src_lang_code, tgt_lang=tgt_lang_code)
        # Tokenize the batch with padding and truncation
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)
        # Generate translations with beam search
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
        # Decode the generated tokens to text
        with tokenizer.as_target_tokenizer():
            decoded_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        # Postprocess the decoded translations
        translations += INDIC_PROCESSOR.postprocess_batch(decoded_tokens, lang=tgt_lang_code)
        del inputs
        clear_cache()  # Clear GPU cache to manage memory usage
    return translations


def translate_text(input_text: str, source_language: str, target_language: str, model, tokenizer) -> str | None:
    """
    Translates the input text from English to a specified target language.
    
    This function validates the source and target languages, splits the input text into sentences,
    translates them in batch, and then joins the translated sentences into a single string.
    
    Args:
        input_text (str): The text to translate.
        source_language (str): Source language (only 'english' is supported).
        target_language (str): Target language (must be one of the allowed target languages).
        model: The translation model.
        tokenizer: The tokenizer for the model.
        
    Returns:
        str | None: The translated text if successful; otherwise, None.
    """
    # Validate that the source language is English
    if source_language.lower() != "english":
        print(f"Error: Translation from '{source_language}' is not supported. Only English is allowed as the source language.")
        return None

    # Validate that the target language is supported
    if target_language.lower() not in ALLOWED_TARGET_LANGUAGES:
        print(f"Error: Translation to '{target_language}' is not supported. Allowed target languages are: {', '.join(ALLOWED_TARGET_LANGUAGES)}.")
        return None

    src_lang_code = "eng_Latn"
    tgt_lang_code = TARGET_LANGUAGE_CODES[target_language.lower()]

    # Split input text into sentences
    input_sentences = split_sentences(input_text, src_lang_code)
    # Translate the batch of sentences
    translated_sentences = batch_translate(input_sentences, src_lang_code, tgt_lang_code, model, tokenizer)
    # Join the translated sentences into one text
    return " ".join(translated_sentences)


def translate_to_all(input_texts: list, source_language: str,
                     en_indic_model, en_indic_tokenizer, indic_indic_model, indic_indic_tokenizer) -> dict:
    """
    Translates a list of input texts to all supported target languages.
    
    For Hindi, the English-to-Indic model is used. For other languages, the Indic-to-Indic model is used.
    
    Args:
        input_texts (list): List of texts to translate.
        source_language (str): Source language (only 'english' is supported).
        en_indic_model: Model for English-to-Indic translation.
        en_indic_tokenizer: Tokenizer for the English-to-Indic model.
        indic_indic_model: Model for Indic-to-Indic translation.
        indic_indic_tokenizer: Tokenizer for the Indic-to-Indic model.
        
    Returns:
        dict: A dictionary mapping target languages to lists of translated texts.
    """
    if source_language.lower() != "english":
        print(f"Error: Translation from '{source_language}' is not supported. Only English is allowed as the source language.")
        return {}

    all_translations = {}
    for target_language in ALLOWED_TARGET_LANGUAGES:
        translations_for_lang = []
        src_lang_code = "eng_Latn"
        tgt_lang_code = TARGET_LANGUAGE_CODES[target_language.lower()]

        # Select the appropriate model and tokenizer based on target language
        if tgt_lang_code == "hin_Deva":
            model_to_use = en_indic_model
            tokenizer_to_use = en_indic_tokenizer
        else:
            model_to_use = indic_indic_model
            tokenizer_to_use = indic_indic_tokenizer

        # Translate each input text individually
        for input_text in input_texts:
            input_sentences = split_sentences(input_text, src_lang_code)
            translated_sentences = batch_translate(input_sentences, src_lang_code, tgt_lang_code, model_to_use, tokenizer_to_use)
            if translated_sentences:
                translations_for_lang.append(" ".join(translated_sentences))
            else:
                translations_for_lang.append("")  # Preserve order even if translation fails
        all_translations[target_language] = translations_for_lang
    return all_translations


def process_input(input_string: str, input_file: str) -> list | None:
    """
    Processes the input either from a direct string or a JSON file.
    
    If an input string is provided, it is wrapped in a list for consistent processing.
    If an input file is provided, it is expected to contain a JSON list.
    
    Args:
        input_string (str): The direct input string.
        input_file (str): The path to the JSON file.
        
    Returns:
        list | None: A list of input texts if successful; otherwise, None.
    """
    if input_string:
        return [input_string]  # Wrap in a list for consistency
    elif input_file:
        if os.path.exists(input_file):
            print(f"Processing input from file: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    else:
                        print("Error: JSON file should contain a list of strings.")
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


def save_translations(translations: dict, input_texts: list, output_file: str) -> None:
    """
    Saves the translations to an output file in a formatted manner.
    
    The output includes a timestamp, the target language, and both the original and translated sentences.
    
    Args:
        translations (dict): Dictionary containing translations keyed by target language.
        input_texts (list): List of original input texts.
        output_file (str): Path to the file where translations will be saved.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Translation Output ({timestamp})\n\n")
        for target_lang, translated_texts in translations.items():
            f.write(f"## Target Language: {target_lang.capitalize()}\n")
            for i, translated_text in enumerate(translated_texts):
                f.write(f"### Input Sentence {i+1}\n")
                f.write("```\n")
                f.write(f"{input_texts[i]}\n")
                f.write("```\n\n")
                f.write(f"### Translated Sentence {i+1}\n")
                f.write("```\n")
                f.write(f"{translated_text}\n")
                f.write("```\n\n")
        f.write("# End of Translation Output\n")
    print(f"Translations saved to: {output_file}")


def main():
    """
    Main function for standalone translation inference.
    
    Parses command-line arguments, processes the input, initializes models and tokenizers,
    performs translation (either to a specific target language or to all supported targets),
    prints the translations, and optionally saves the results to an output file.
    """
    parser = argparse.ArgumentParser(
        description="Translate English text to Hindi, Marathi, Gujarati, or Tamil."
    )
    parser.add_argument("--source_language", type=str, default="english",
                        help="Source language (default: english)")
    # Mutually exclusive group for target language vs. translating to all targets
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target_language", type=str,
                       help="Target language (hindi, marathi, gujarati, tamil)")
    group.add_argument("--translate_to_all_targets", action="store_true",
                       help="Translate to all supported target languages")
    # Mutually exclusive group for input type (string or file)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_string", type=str,
                             help="Input string to translate")
    input_group.add_argument("--input_file", type=str,
                             help="Path to a JSON file containing input (list of strings)")
    parser.add_argument("--output_file", type=str,
                        help="Path to save the output file (optional)")
    parser.add_argument("--lora_adapter_dir", type=str,
                        help="Path to the LoRA adapter directory (optional)")
    parser.add_argument("--quantization", type=str, choices=["4-bit", "8-bit", ""], default="",
                        help="Enable quantization.")
    parser.add_argument("--attn_implementation", type=str, choices=["flash_attention_2", "eager"],
                        default="flash_attention_2",
                        help="Attention implementation to use.")

    args = parser.parse_args()

    # Process input texts from either direct string or file
    input_texts = process_input(args.input_string, args.input_file)
    if not input_texts:
        return

    # Initialize models and tokenizers for both translation paths
    # For Hindi, use the English-to-Indic model; for other languages, use the Indic-to-Indic model.
    en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(
        EN_INDIC_CKPT_DIR,
        quantization=args.quantization,
        attn_implementation=args.attn_implementation,
        lora_adapter_dir=args.lora_adapter_dir if args.target_language and args.target_language.lower() == "hindi" else None
    )
    indic_indic_tokenizer, indic_indic_model = initialize_model_and_tokenizer(
        INDIC_INDIC_CKPT_DIR,
        quantization=args.quantization,
        attn_implementation=args.attn_implementation,
        lora_adapter_dir=args.lora_adapter_dir if args.target_language and args.target_language.lower() != "hindi" else None
    )

    # Translate texts to all supported target languages if requested
    if args.translate_to_all_targets:
        translations = translate_to_all(
            input_texts, args.source_language,
            en_indic_model, en_indic_tokenizer,
            indic_indic_model, indic_indic_tokenizer
        )
        print("\n--- Translations to All Target Languages ---")
        for target_lang, translated_texts in translations.items():
            print(f"\nTarget Language: {target_lang.capitalize()}")
            for i, translated_text in enumerate(translated_texts):
                print(f"Input Sentence {i+1}:")
                print("```\n" + input_texts[i] + "\n```")
                print(f"Translated Sentence {i+1}:")
                print("```\n" + translated_text + "\n```")
        # Save translations to file if output_file is provided
        if args.output_file:
            save_translations(translations, input_texts, args.output_file)
    elif args.target_language:
        # Translate to a specific target language
        all_translations = {}
        translated_texts = []
        # Choose model and tokenizer based on the target language
        model_to_use = en_indic_model if args.target_language.lower() == "hindi" else indic_indic_model
        tokenizer_to_use = en_indic_tokenizer if args.target_language.lower() == "hindi" else indic_indic_tokenizer
        for input_text in input_texts:
            translated_text = translate_text(input_text, args.source_language, args.target_language, model_to_use, tokenizer_to_use)
            translated_texts.append(translated_text if translated_text else None)
        all_translations[args.target_language] = translated_texts

        print(f"\n--- Translation to {args.target_language.capitalize()} ---")
        for i, translated_text in enumerate(translated_texts):
            print(f"Input Sentence {i+1}:")
            print("```\n" + input_texts[i] + "\n```")
            print(f"Translated Sentence {i+1}:")
            print("```\n" + str(translated_text) + "\n```")

        # Save output to file if specified
        if args.output_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(f"# Translation Output ({timestamp})\n\n")
                f.write(f"## Target Language: {args.target_language.capitalize()}\n")
                for i, translated_text in enumerate(translated_texts):
                    f.write(f"### Input Sentence {i+1}\n")
                    f.write("```\n" + input_texts[i] + "\n```\n\n")
                    f.write(f"### Translated Sentence {i+1}\n")
                    f.write("```\n" + str(translated_text) + "\n```\n\n")
                f.write("# End of Translation Output\n")
            print(f"Translations saved to: {args.output_file}")


if __name__ == "__main__":
    # Standalone initialization for inference
    print("Initializing models and tokenizers for standalone inference (this will not run if the module is imported)...")
    en_indic_tokenizer_global, en_indic_model_global = initialize_model_and_tokenizer(
        EN_INDIC_CKPT_DIR, quantization="", attn_implementation="flash_attention_2"
    )
    indic_indic_tokenizer_global, indic_indic_model_global = initialize_model_and_tokenizer(
        INDIC_INDIC_CKPT_DIR, quantization="", attn_implementation="flash_attention_2"
    )
    print("Models and tokenizers initialized for standalone inference.")
    main()
