import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import yaml
import logging
from utils.logger import logger

def load_config():
    """
    Load configuration from config.yaml.
    """
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config.yaml: {e}")
        return {}

def generate_product_listing(attributes):
    """
    Generate SEO-optimized product listing using GPT-2.
    """
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        prompt = f"Introducing a {attributes['color']} {attributes['type']} made of {attributes['material']} from {attributes['brand']}. "
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        outputs = model.generate(
            inputs,
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        listing = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return listing
    except Exception as e:
        logger.error(f"Error generating product listing: {e}")
        return ""

def main():
    """
    Example usage of the text embedding model.
    """
    config = load_config()
    sample_attributes = {
        'type': 'ring',
        'material': 'gold',
        'color': 'yellow',
        'brand': 'BrandX'
    }
    listing = generate_product_listing(sample_attributes)
    print(listing)

if __name__ == "__main__":
    main()
