import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def calculate_perplexity(text):
    # Load pre-trained model (weights)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Encode text
    encodings = tokenizer(text, return_tensors='pt')

    # Calculate loss
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    
    return perplexity.item()

# Example usage
text = "A talking screwdriver"
text1 = "A "
text2 = "A talking"
text3 = "The man was talking Platt deutsch"

perplexity1 = calculate_perplexity(text1)
perplexity2 = calculate_perplexity(text2)
perplexity3 = calculate_perplexity(text3)
perplexity = calculate_perplexity(text)
print(f'Perplexity: {perplexity}')
print(f'Perplexity1: {perplexity1}')
print(f'Perplexity2: {perplexity2}')
print(f'Perplexity3: {perplexity3}')

