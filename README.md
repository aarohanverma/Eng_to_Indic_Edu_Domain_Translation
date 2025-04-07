
# 📝 English-to-Indic Translation Pipeline

This repository provides an end-to-end pipeline for fine-tuning and using the [ai4bharat/indictrans2-en-indic-1B](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B) model with LoRA adapters for English to Indic translation. It includes scripts for setup, training, inference, and evaluation.

---

## ⚙️ Virtual Environment Setup

It is highly recommended to create and activate a virtual environment to isolate the project dependencies and avoid conflicts with your system-wide Python packages.

### Steps to create and activate a virtual environment:

1.  **Create a virtual environment:** Open your terminal in the repository's root directory and run the following command. You can choose any name for your environment (e.g., `venv`, `.venv`, `env`).

    ```bash
    python3 -m venv venv
    # or if python3 is not your default python command
    python -m venv venv
    ```

2.  **Activate the virtual environment:**


    ```bash
    source venv/bin/activate
    ```




    Once activated, you will see the name of your virtual environment in parentheses at the beginning of your terminal prompt (e.g., `(venv) user@host:~/your_repo$`).

---

## 📦 1. Setup

To initialize the environment and install all dependencies, run:

```bash
bash setup.sh
```

### What `setup.sh` does:

- Unzips dependency archives from the `dependencies/` folder
- Installs:
  - `indic_nlp_library`
  - `IndicTransToolkit` with its requirements
  - Additional requirements from `dependencies/requirements.txt`
- Compiles Cython extensions for IndicTransToolkit
- Downloads required NLTK tokenizer data
- Executes `IndicTrans2/huggingface_interface/install.sh`
- Reinstalls `IndicTransToolkit` to reflect local changes

---

## 🏋️‍♂️ 2. Training

Train LoRA adapters by executing:

```bash
bash train.sh [OPTIONS]
```

### Directory Structure (REQUIRED):

```
en-indic-exp/
├── train/
│   ├── eng_Latn-hin_Deva/
│   │   ├── train.eng_Latn
│   │   └── train.hin_Deva
│   ├── eng_Latn-tam_Taml/
│   │   └── ...
│   └── {src_lang}-{tgt_lang}/
│       ├── train.{src_lang}
│       └── train.{tgt_lang}
└── dev/
    ├── eng_Latn-hin_Deva/
    │   ├── dev.eng_Latn
    │   └── dev.hin_Deva
    ├── eng_Latn-tam_Taml/
    │   └── ...
    └── {src_lang}-{tgt_lang}/
        ├── dev.{src_lang}
        └── dev.{tgt_lang}
```

### Key Parameters:

| Flag               | Description                                      | Default                                   |
|--------------------|--------------------------------------------------|-------------------------------------------|
| `-d <dir>`         | Path to training dataset                         | `datasets/train_validation/en-indic-exp`  |
| `-m <model_name>`  | Base model name                                  | `ai4bharat/indictrans2-en-indic-1B`       |
| `-o <dir>`         | Output directory for LoRA adapters               | `lora_adapters/output_<timestamp>`        |
| `--src <langs>`    | Comma-separated source language codes            | `eng_Latn`                                |
| `--tgt <langs>`    | Comma-separated target language codes            | `guj_Gujr,hin_Deva,mar_Deva,tam_Taml`     |
| `-h, --help`       | Show help message                                |                                           |

**Other advanced training hyperparameters are set in the script and can be modified if needed.**

---

## 🌍 3. Inference

Use the following command to generate translations:

```bash
bash translate.sh [OPTIONS]
```

### Usage Scenarios:

```bash
# Translate a string to Hindi using base model
bash translate.sh -t hindi -s "Welcome!"

# Translate using LoRA model
bash translate.sh -m lora --lora_adapter_dir path/to/lora -t marathi -s "How are you?"

# Translate to all supported languages
bash translate.sh -a -s "Namaste!"

# Translate using input file and write results to file
bash translate.sh -t gujarati -f test_data.json -o output.txt
```

### Key Parameters:

| Flag                       | Description                                                    |
|----------------------------|----------------------------------------------------------------|
| `-m <base/lora>`           | Use base model or LoRA-adapted model                           |
| `--lora_adapter_dir <dir>` | Directory for LoRA adapter (required if model is `lora`)       |
| `-t <lang>`                | Target language (hindi, marathi, gujarati, tamil)              |
| `-a`                       | Translate to all supported languages                           |
| `-s <text>`                | Text string to translate                                       |
| `-f <file>`                | JSON file with list of strings                                 |
| `-o <file>`                | Output file to save translations                               |
| `-h, --help`               | Show help                                                      |

### Inference JSON Format:

```json
[
  "This is first sentence.",
  "This is second sentence."
]
```

---

## 📊 4. Evaluation

Evaluate model outputs using chrF score:

```bash
bash evaluation.sh [OPTIONS]
```

### Key Parameters:

| Flag                                          | Description                                          |
|-----------------------------------------------|------------------------------------------------------|
| `-m, --model_type <base/lora>`                | Use base or LoRA model                               |
| `-l, --lora_adapter_dir <dir>`                | Path to LoRA adapter (required if model is `lora`)   |
| `-i, --input_file <path>`                     | Path to evaluation data JSON (REQUIRED)              |
| `-q, --quantization <4-bit/8-bit>`            | Quantization mode (optional)                         |
| `-a, --attention <flash_attention_2/eager>`   | Attention implementation (optional)                  |
| `-h, --help`                                  | Show help                                            |

### Evaluation JSON Format:

```json
[
  {
    "english": "Where is the library?",
    "hindi": "पुस्तकालय कहाँ है?",
    "marathi": "ग्रंथालय कुठे आहे?",
    "gujarati": "લાઇબ્રેરી ક્યાં છે?",
    "tamil": "நூலகம் எங்கே?"
  },
  ...
]
```



- The `english` field is used as model input.
- Target fields (e.g., `hindi`, `tamil`) contain **gold-standard reference translations**.
- Model predictions are compared with references using the **chrF** metric.
- The closer the generated output is to human reference, the higher the score.

---

## ⚠️ Troubleshooting

### Windows-style Line Endings Error

If you encounter errors like:

```bash
setup.sh: line 15: $'\r': command not found
setup.sh: line 17: $'\r': command not found
...
```

This happens due to **Windows (CRLF)** line endings. Convert the file to **Unix (LF)** using:

```bash
sudo apt-get install dos2unix  # install if not already installed
dos2unix setup.sh              # convert script
```

Output:

```bash
dos2unix: converting file setup.sh to Unix format...
```

Repeat for other scripts if needed.

---

## 📌 Notes

- Make `.sh` scripts executable:

```bash
chmod +x *.sh
```

- Customize training and evaluation flags to suit your data or experimentation setup.
- All scripts are modular and can be extended as needed.

---
