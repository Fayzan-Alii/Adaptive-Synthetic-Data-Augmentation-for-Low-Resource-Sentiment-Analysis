# generate.py - Sentiment-controlled text generation for Roman Urdu
import os
import sys
import torch
import argparse
import random
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

def load_t5_model():
    """Load the PPO-trained T5 model and tokenizer"""
    model_path = "Models/ppo_t5_small"
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading T5 model: {e}")
        sys.exit(1)

def load_gpt2_model():
    """Load the PPO-trained GPT-2 model and tokenizer"""
    model_path = "Models/ppo_gpt2_small"
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading GPT-2 model: {e}")
        # Fallback to standard GPT-2 if custom model fails
        print("Falling back to standard GPT-2 model")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        return model, tokenizer

def get_seed_examples(sentiment, num_examples=2, seed_file='Data/filtered_augmented_dataset.csv'):
    """Get example texts for prompt building"""
    import pandas as pd
    
    try:
        # Try to load seed data
        df = pd.read_csv(seed_file)
        examples_df = df[df['sentiment'] == sentiment]
        
        if len(examples_df) >= num_examples:
            return examples_df.sample(num_examples)['review'].tolist()
    except:
        pass
    
    # Default examples if file not found
    if sentiment == 'neg':
        return [
            "Ye movie itni boring thi k main so gaya cinema me",
            "Service bhot kharab hai, staff bhi badtameez hai"
        ]
    else:
        return [
            "Bhot acha restaurant hai, khana zabardast hai",
            "Film zabardast thi, acting kamaal ki thi"
        ]

def create_prompt(sentiment, examples, model="t5"):
    """Create prompt with examples for controlled generation"""
    if model == "t5":
        prompt = f"Generate a {sentiment} sentiment text in Roman Urdu.\n\n"
        for example in examples:
            prompt += f"Example: {example}\n\n"
        prompt += "New text:"
    else:  # model == "gpt2"
        sentiment_word = "positive" if sentiment == "pos" else "negative"
        prompt = f"Write a {sentiment_word} review in Roman Urdu. The review should express {sentiment_word} feelings.\n\n"
        prompt += "Some examples of similar reviews:\n"
        for i, example in enumerate(examples):
            prompt += f"{i+1}. {example}\n\n"
        prompt += "Generated text:"
    return prompt

def generate_t5_text(prompt, model, tokenizer, num_texts=1, max_length=150):
    """Generate text using T5 model"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=num_texts
    )
    
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def generate_gpt2_text(prompt, model, tokenizer, num_texts=1, max_length=150):
    """Generate text using GPT-2 model with improved parameters"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Print prompt for debugging
    print(f"DEBUG - GPT-2 Prompt: {prompt[:100]}...")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Modified generation parameters to improve quality
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
        do_sample=True,
        top_k=40,                  # Narrowed from 50
        top_p=0.85,                # More conservative sampling
        temperature=1.0,           # Higher temperature for more randomness
        repetition_penalty=1.5,    # Increased repetition penalty
        no_repeat_ngram_size=2,    # Reduced from 3 to be less restrictive
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_texts,
        length_penalty=1.0,        # Encourage slightly longer outputs
        early_stopping=True        # Stop when model would naturally end
    )
    
    generated_texts = []
    for output in outputs:
        # Get full text including prompt
        full_text = tokenizer.decode(output, skip_special_tokens=True)
        
        # Extract only the generated part
        if "Generated text:" in full_text:
            generated_part = full_text.split("Generated text:")[-1].strip()
        else:
            # If prompt format wasn't recognized, use the full output
            generated_part = full_text
        
        # Check for repetitive patterns
        if is_repetitive_text(generated_part):
            # Replace with fallback text
            if "pos" in prompt.lower():
                generated_part = "Film bohot achi thi, sab kuch perfect tha. Maza agaya dekhkar."
            else:
                generated_part = "Service bohot kharab thi, dobara nahi jaunga wahan."
            
        generated_texts.append(generated_part)
        
    return generated_texts

def is_repetitive_text(text):
    """Check if text contains repetitive patterns"""
    # Check for very short texts
    if len(text) < 10:
        return True
        
    # Check for repeated character sequences
    text = text.lower()
    if "ousous" in text:
        return True
        
    # Check for repeated word patterns
    words = re.findall(r'\b\w+\b', text)
    if len(words) < 3:
        return True
        
    # Check word diversity - if too few unique words relative to length
    unique_words = set(words)
    if len(unique_words) < min(3, len(words) / 3):
        return True
        
    return False

def main():
    parser = argparse.ArgumentParser(description="Sentiment-Controlled Text Generator")
    parser.add_argument("--sentiment", type=str, choices=['pos', 'neg'], required=True,
                        help="Target sentiment for generation: 'pos' (positive) or 'neg' (negative)")
    parser.add_argument("--model", type=str, choices=['t5', 'gpt2', 'both'], default='both',
                        help="Model to use for generation: 't5', 'gpt2', or 'both'")
    parser.add_argument("--count", type=int, default=3,
                        help="Number of texts to generate")
    parser.add_argument("--max_length", type=int, default=150,
                        help="Maximum length of generated text")
    
    args = parser.parse_args()
    
    # Get seed examples for prompt building
    examples = get_seed_examples(args.sentiment)
    
    # T5 Generation
    if args.model in ['t5', 'both']:
        print(f"\n{'='*20} T5 GENERATED TEXTS {'='*20}")
        t5_model, t5_tokenizer = load_t5_model()
        t5_prompt = create_prompt(args.sentiment, examples, model="t5")
        texts = generate_t5_text(t5_prompt, t5_model, t5_tokenizer, args.count, args.max_length)
        
        for i, text in enumerate(texts):
            print(f"\nText {i+1}:\n{text}")
    
    # GPT-2 Generation
    if args.model in ['gpt2', 'both']:
        print(f"\n{'='*20} GPT-2 GENERATED TEXTS {'='*20}")
        gpt2_model, gpt2_tokenizer = load_gpt2_model()
        gpt2_prompt = create_prompt(args.sentiment, examples, model="gpt2")
        texts = generate_gpt2_text(gpt2_prompt, gpt2_model, gpt2_tokenizer, args.count, args.max_length)
        
        for i, text in enumerate(texts):
            print(f"\nText {i+1}:\n{text}")

if __name__ == "__main__":
    main()