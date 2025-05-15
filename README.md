# Roman Urdu Sentiment-Controlled Text Generation

## Overview
This project provides a complete pipeline for sentiment-controlled text generation in Roman Urdu using advanced NLP models. It leverages T5 and GPT-2 architectures, fine-tuned and further optimized with Proximal Policy Optimization (PPO) for generating high-quality, sentiment-specific text. The project also includes robust sentiment classification using BERT and XLM-RoBERTa models, and supports reproducible deployment via Docker.

---

## Features
- **Sentiment-Controlled Generation**: Generate Roman Urdu text with positive or negative sentiment using T5 and GPT-2 models.
- **Custom Training & PPO Fine-Tuning**: Models are fine-tuned on a curated Roman Urdu dataset and further improved with reinforcement learning (PPO).
- **Sentiment Classification**: BERT and XLM-RoBERTa models for accurate sentiment analysis and evaluation.
- **Data Augmentation**: Includes scripts and datasets for augmenting and filtering Roman Urdu reviews.
- **Dockerized Deployment**: Easily build and run the project in a reproducible environment, with GPU support.
- **Jupyter Notebook**: End-to-end code for data processing, training, evaluation, and analysis.

---

## Project Structure
```
project/
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── generate.py                # Main script for sentiment-controlled text generation
├── GenAI_Project.ipynb        # Notebook: data prep, training, evaluation, analysis
├── Data/                      # Datasets (CSV, TSV)
├── Images/                    # Plots and visualizations
├── Models/                    # All model checkpoints and tokenizers
│   ├── ppo_t5_small/
│   ├── ppo_gpt2_small/
│   ├── bert_finetuned/
│   ├── xlm-roberta-finetuned/
│   └── ...
└── ...
```

---

## Installation
### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (optional, for faster inference/training)
- Docker (for containerized deployment)

### Local Setup
1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd project
   ```
2. **Install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Download/prepare models**
   - Place trained model folders (e.g., `ppo_t5_small`, `ppo_gpt2_small`) in `Models/`.
   - Place datasets in `Data/`.

---

## Usage
### Text Generation (Script)
Generate Roman Urdu text with controlled sentiment:
```bash
python generate.py --sentiment pos --model both --count 3
```
- `--sentiment`: `pos` (positive) or `neg` (negative)
- `--model`: `t5`, `gpt2`, or `both`
- `--count`: Number of texts to generate
- `--max_length`: Maximum output length (default: 150)

### Example Output
```
==================== T5 GENERATED TEXTS ====================
Text 1:
Bhot acha restaurant hai, khana zabardast hai...

==================== GPT-2 GENERATED TEXTS ====================
Text 1:
Film bohot achi thi, sab kuch perfect tha. Maza agaya dekhkar.
```

### Jupyter Notebook
- Open `GenAI_Project.ipynb` for full data processing, training, evaluation, and analysis code.

---

## Docker Deployment
### Build Docker Image
```bash
docker build -t roman-urdu-genai .
```

### Run Container (CPU)
```bash
docker run --rm -it \
  -v $(pwd)/Models:/app/Models \
  -v $(pwd)/Data:/app/Data \
  roman-urdu-genai \
  python generate.py --sentiment pos --model both --count 3
```

### Run Container (GPU, with NVIDIA Docker)
```bash
docker run --rm -it --gpus all \
  -v $(pwd)/Models:/app/Models \
  -v $(pwd)/Data:/app/Data \
  roman-urdu-genai \
  python generate.py --sentiment neg --model gpt2 --count 2
```

**Note:**
- Mount `Models/` and `Data/` as volumes to avoid copying large files into the image.
- For custom models, update the `model_path` in `generate.py` if needed.

---

## Technical Details
### Models
- **T5 & GPT-2**: Fine-tuned on Roman Urdu reviews, further optimized with PPO for sentiment control.
- **BERT & XLM-RoBERTa**: Used for sentiment classification and evaluation. Fine-tuned on the same dataset.
- **fastText**: Used for baseline experiments and data augmentation.

### Data
- Datasets are in `Data/` (CSV/TSV). Includes original, augmented, and filtered datasets.
- Example files: `augmented_dataset.csv`, `filtered_enhanced_dataset.csv`, etc.

### Evaluation
- Metrics: Accuracy, F1-score (for classifiers), and qualitative analysis for generated text.
- Visualizations: See `Images/` for confusion matrices, quality metrics, and training progress.

---

## Customization & Extension
- **Training**: Use the notebook to retrain or fine-tune models on new data.
- **Prompt Engineering**: Modify `generate.py` to change prompt templates or generation parameters.
- **Evaluation**: Extend evaluation scripts for new metrics or human evaluation.

---

## References
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [trl (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [PyTorch](https://pytorch.org/)
- [fastText](https://fasttext.cc/)

---

## Contact
For questions or collaboration, please contact the project maintainer.
