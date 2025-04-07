import os
import argparse
import random
import gc
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sacrebleu.metrics import CHRF
from peft import LoraConfig, get_peft_model
from IndicTransToolkit.processor import IndicProcessor
from IndicTransToolkit.collator import IndicDataCollator

# Import necessary classes from transformers
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def clear_cache():
    """
    Clears the Python garbage collector and the CUDA cache (if available)
    to free up memory before starting training.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"Failed to clear CUDA cache: {e}")


def set_seed(seed: int):
    """
    Set seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior in cuDNN (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Set a fixed seed for reproducibility
set_seed(42)

# Initialize the chrF metric from sacreBLEU
chrf_metric = CHRF()


def get_arg_parse():
    """
    Creates and returns the argument parser for command-line arguments.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Train LoRA adapters for translation models.")
    parser.add_argument("--model", type=str, required=True, help="Pre-trained model name or path.")
    parser.add_argument(
        "--src_lang_list",
        type=str,
        required=True,
        help="Comma separated list of source languages."
    )
    parser.add_argument(
        "--tgt_lang_list",
        type=str,
        required=True,
        help="Comma separated list of target languages."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Number of steps between saving checkpoints.")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps between evaluations.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device for training/evaluation.")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs.")
    parser.add_argument("--max_steps", type=int, default=1000000, help="Total number of training steps to perform.")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("--warmup_steps", type=int, default=4000, help="Number of warmup steps for learning rate scheduler.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for training.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay coefficient.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 parameter for Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.98, help="Beta2 parameter for Adam optimizer.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for the model.")
    parser.add_argument("--print_samples", action="store_true", help="Print sample predictions during evaluation.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw_torch",
        choices=["adam_hf", "adamw_torch", "adamw_torch_fused", "adamw_apex_fused", "adafactor"],
        help="Optimizer to use for training."
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="inverse_sqrt",
        choices=["inverse_sqrt", "linear", "polynomial", "cosine", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type."
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of subprocesses for data loading.")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss", help="Metric for selecting the best model.")
    parser.add_argument("--greater_is_better", action="store_true", help="Whether a higher metric value is better.")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj", help="Comma separated list of modules to target with LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA layers.")
    parser.add_argument("--lora_r", type=int, default=16, help="Rank for LoRA layers.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Scaling factor for LoRA layers.")
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["wandb", "tensorboard", "azure_ml", "none"],
        help="Reporting tool to use for logging."
    )
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (number of evaluations with no improvement).")
    parser.add_argument("--threshold", type=float, default=1e-3, help="Early stopping threshold for improvement.")
    return parser


def load_and_process_translation_dataset(
    data_dir,
    split="train",
    tokenizer=None,
    processor=None,
    src_lang_list=None,
    tgt_lang_list=None,
    num_proc=2,
    seed=42
):
    """
    Loads and processes a translation dataset from a given directory.
    For each source-target language pair, the function reads the corresponding
    text files, applies preprocessing, and combines them into a single dataset.

    Args:
        data_dir (str): The directory where the dataset files are stored.
        split (str): Dataset split to load (e.g., 'train' or 'dev').
        tokenizer: The tokenizer instance for encoding the text.
        processor: An instance of IndicProcessor for preprocessing.
        src_lang_list (list): List of source language codes.
        tgt_lang_list (list): List of target language codes.
        num_proc (int): Number of processes for dataset mapping.
        seed (int): Random seed for shuffling.

    Returns:
        Dataset: A Hugging Face Dataset object containing preprocessed examples.
    """
    complete_dataset = {
        "sentence_SRC": [],
        "sentence_TGT": [],
    }

    # Loop through every source-target language pair (skip pairs with identical languages)
    for src_lang in src_lang_list:
        for tgt_lang in tgt_lang_list:
            if src_lang == tgt_lang:
                continue
            src_path = os.path.join(data_dir, split, f"{src_lang}-{tgt_lang}", f"{split}.{src_lang}")
            tgt_path = os.path.join(data_dir, split, f"{src_lang}-{tgt_lang}", f"{split}.{tgt_lang}")
            if not os.path.exists(src_path) or not os.path.exists(tgt_path):
                raise FileNotFoundError(
                    f"Source ({split}.{src_lang}) or Target ({split}.{tgt_lang}) file not found in {data_dir}"
                )
            with open(src_path, encoding="utf-8") as src_file, open(tgt_path, encoding="utf-8") as tgt_file:
                src_lines = src_file.readlines()
                tgt_lines = tgt_file.readlines()

            # Check that both source and target files have the same number of lines
            if len(src_lines) != len(tgt_lines):
                raise AssertionError(
                    f"Source and target files have different number of lines for {split}.{src_lang} and {split}.{tgt_lang}"
                )

            # Preprocess the source and target lines using the processor
            complete_dataset["sentence_SRC"] += processor.preprocess_batch(
                src_lines, src_lang=src_lang, tgt_lang=tgt_lang, is_target=False
            )
            complete_dataset["sentence_TGT"] += processor.preprocess_batch(
                tgt_lines, src_lang=tgt_lang, tgt_lang=src_lang, is_target=True
            )

    # Convert the dictionary to a Hugging Face Dataset and shuffle it with a fixed seed
    dataset = Dataset.from_dict(complete_dataset).shuffle(seed=seed)

    # Map the dataset using the tokenizer via the preprocess_fn function
    return dataset.map(
        lambda example: preprocess_fn(example, tokenizer=tokenizer),
        batched=True,
        num_proc=num_proc,
    )


def compute_metrics_factory(tokenizer, metric_dict=None, print_samples=False, n_samples=10):
    """
    Factory function to create a compute_metrics function for evaluation.

    Args:
        tokenizer: The tokenizer used for decoding predictions and labels.
        metric_dict (dict): Dictionary mapping metric names to metric objects.
        print_samples (bool): Whether to print sample predictions.
        n_samples (int): Number of samples to print if print_samples is True.

    Returns:
        function: A function that computes and returns the evaluation metrics.
    """
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Replace label padding tokens with the tokenizer's pad token ID
        labels[labels == -100] = tokenizer.pad_token_id
        preds[preds == -100] = tokenizer.pad_token_id

        # Decode predictions and labels
        with tokenizer.as_target_tokenizer():
            preds_decoded = [
                x.strip() for x in tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            ]
            labels_decoded = [
                x.strip() for x in tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            ]

        if len(preds_decoded) != len(labels_decoded):
            raise AssertionError("Predictions and labels have different lengths")

        # Optionally, print a random sample of predictions and references
        sample_df = pd.DataFrame({"Predictions": preds_decoded, "References": labels_decoded}).sample(n=n_samples)
        if print_samples:
            for pred, label in zip(sample_df["Predictions"].values, sample_df["References"].values):
                print(f" | > Prediction: {pred}")
                print(f" | > Reference: {label}\n")

        # Compute metrics using the provided metric objects
        return {
            metric_name: metric.corpus_score(preds_decoded, [labels_decoded]).score
            for metric_name, metric in metric_dict.items()
        }

    return compute_metrics


def preprocess_fn(example, tokenizer, **kwargs):
    """
    Preprocess a single example by tokenizing the source and target sentences.

    Args:
        example (dict): A dictionary with keys 'sentence_SRC' and 'sentence_TGT'.
        tokenizer: The tokenizer to encode the text.
    
    Returns:
        dict: A dictionary with tokenized inputs and labels.
    """
    # Tokenize the source sentence
    model_inputs = tokenizer(
        example["sentence_SRC"], truncation=True, padding=False, max_length=256
    )
    # Tokenize the target sentence within the target tokenizer context
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["sentence_TGT"], truncation=True, padding=False, max_length=256
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(args):
    """
    Main function to orchestrate the training process.

    Args:
        args: Parsed command-line arguments.
    """
    # Clear caches to free memory
    clear_cache()
    print(f" | > Loading model '{args.model}' and tokenizer ...")
    
    # Load pre-trained model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        attn_implementation="eager",
        dropout=args.dropout
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Initialize the IndicProcessor for preprocessing (in non-inference mode)
    processor = IndicProcessor(inference=False)
    
    # Prepare the data collator for dynamic padding during training/evaluation
    data_collator = IndicDataCollator(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )

    # Load and preprocess the training and evaluation datasets.
    if args.data_dir is not None:
        # Load training dataset
        train_dataset = load_and_process_translation_dataset(
            args.data_dir,
            split="train",
            tokenizer=tokenizer,
            processor=processor,
            src_lang_list=args.src_lang_list.split(","),
            tgt_lang_list=args.tgt_lang_list.split(","),
        )
        print(f" | > Loaded train dataset from {args.data_dir}. Size: {len(train_dataset)}")

        # Load evaluation dataset
        eval_dataset = load_and_process_translation_dataset(
            args.data_dir,
            split="dev",
            tokenizer=tokenizer,
            processor=processor,
            src_lang_list=args.src_lang_list.split(","),
            tgt_lang_list=args.tgt_lang_list.split(","),
        )
        print(f" | > Loaded eval dataset from {args.data_dir}. Size: {len(eval_dataset)}")
    else:
        raise ValueError(" | > Data directory not provided")

    # Configure LoRA parameters using PEFT.
    lora_config = LoraConfig(
        r=args.lora_r,
        bias="none",
        inference_mode=False,
        task_type="SEQ_2_SEQ_LM",
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
    )

    # Set label smoothing for the model (if supported)
    model.set_label_smoothing(args.label_smoothing)

    # Wrap the model with PEFT to apply LoRA adaptations
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Setup the metric computation function using chrF.
    print(" | > Initializing metric computation with chrF ...")
    seq2seq_compute_metrics = compute_metrics_factory(
        tokenizer=tokenizer,
        print_samples=args.print_samples,
        metric_dict={"chrF": chrf_metric},
    )

    # Define training arguments for Seq2SeqTrainer.
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        fp16=True,
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=100,
        save_total_limit=1,
        predict_with_generate=True,
        load_best_model_at_end=True,
        max_steps=args.max_steps,  # max_steps takes precedence over num_train_epochs
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_accumulation_steps=args.grad_accum_steps,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
        optim=args.optimizer,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.num_workers,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        report_to=args.report_to,
        generation_max_length=256,
        generation_num_beams=5,
        sortish_sampler=True,
        group_by_length=True,
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True,
        dataloader_prefetch_factor=2,
        bf16=False
    )

    # Initialize the Trainer with the model, datasets, data collator, and metrics.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=seq2seq_compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.patience,
                early_stopping_threshold=args.threshold,
            )
        ],
    )

    print(" | > Starting training ...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print(" | > Training interrupted by user...")

    # Save only the LoRA adapter weights to the output directory.
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = get_arg_parse()
    args = parser.parse_args()
    main(args)
