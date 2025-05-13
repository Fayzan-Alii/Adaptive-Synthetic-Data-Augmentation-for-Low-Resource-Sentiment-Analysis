# inference.py - Sentiment Analysis for Roman Urdu Text
import os
import sys
import torch
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_path):
    """Load the trained model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment of input text"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.softmax(outputs.logits, dim=1)
        sentiment_id = torch.argmax(prediction, dim=1).item()
    
    return "Positive" if sentiment_id == 1 else "Negative", prediction[0][sentiment_id].item()

def get_available_models():
    """Return list of available pre-trained models"""
    models = [
        "Models/xlm-roberta-finetuned",      # T5-synthetic data trained XLM-RoBERTa
        "Models/xlm-roberta-finetuned-gpt2", # GPT2-synthetic data trained XLM-RoBERTa
        "Models/bert_finetuned",             # T5-synthetic data trained BERT
        "Models/bert_finetuned_gpt2"         # GPT2-synthetic data trained BERT
    ]
    
    available_models = []
    for model in models:
        if os.path.exists(model):
            available_models.append(model)
    
    return available_models

def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Inference")
    
    # Get available models
    available_models = get_available_models()
    default_model = available_models[0] if available_models else "Models/xlm-roberta-finetuned-gpt2"
    
    # Model selection
    parser.add_argument("--model_path", type=str, default=default_model,
                        help=f"Path to the fine-tuned model. Available: {', '.join(available_models)}")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to analyze")
    parser.add_argument("--list_models", action="store_true", 
                        help="List all available models")
    parser.add_argument("--compare", action="store_true",
                        help="Compare predictions from all available models")
    
    args = parser.parse_args()
    
    # List available models and exit
    if args.list_models:
        print("Available models:")
        for model in available_models:
            print(f"  - {model}")
        return
        
    # Compare all models
    if args.compare:
        print(f"Comparing predictions for: {args.text}")
        print("-" * 50)
        
        for model_path in available_models:
            print(f"Loading {model_path}...")
            model, tokenizer = load_model(model_path)
            sentiment, confidence = predict_sentiment(args.text, model, tokenizer)
            model_type = "BERT" if "bert" in model_path.lower() else "XLM-RoBERTa"
            data_type = "GPT-2" if "gpt2" in model_path.lower() else "T5"
            print(f"{model_type} ({data_type} data): {sentiment} (confidence: {confidence:.4f})")
        return
    
    # Standard single model prediction
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path)
    
    print(f"Running inference on: {args.text}")
    sentiment, confidence = predict_sentiment(args.text, model, tokenizer)
    
    print(f"Result: {sentiment} (confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()