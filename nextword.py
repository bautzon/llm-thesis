import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import numpy as np
import math

# Load pre-trained model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Encode the input text
input_text = "Barack"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Get the model's predictions
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs.logits

# Get the probabilities for the next word
next_token_logits = predictions[:, -1, :]
probabilities = torch.softmax(next_token_logits, dim=-1).squeeze()

# Compute entropy
entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9)).item()  # Adding a small value to avoid log(0)
print(f"Entropy: {entropy}")

# Convert to numpy for plotting
probabilities_np = probabilities.numpy()
top_k = 10  # Number of top probabilities to plot
top_k_probs, top_k_indices = torch.topk(probabilities, top_k)

# Get the words corresponding to the top_k indices
top_k_words = [tokenizer.decode([idx]) for idx in top_k_indices]

# Plot the probabilities
plt.figure(figsize=(10, 6))
plt.bar(top_k_words, top_k_probs.numpy())
plt.xlabel('Words')
plt.ylabel('Probability')
plt.title(f'Top {top_k} Probabilities for the Next Word after "Barack"')
plt.show()