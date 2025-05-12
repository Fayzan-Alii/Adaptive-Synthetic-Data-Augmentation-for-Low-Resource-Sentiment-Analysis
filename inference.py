# inference.py
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

def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Inference")
    parser.add_argument("--model_path", type=str, default="Models/xlm-roberta-finetuned-gpt2",
                        help="Path to the fine-tuned model")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to analyze")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path)
    
    print(f"Running inference on: {args.text}")
    sentiment, confidence = predict_sentiment(args.text, model, tokenizer)
    
    print(f"Result: {sentiment} (confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()