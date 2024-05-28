import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Example text and encoding
text = "This is a test sentence for visualizing perplexity."
encodings = tokenizer(text, return_tensors='pt', padding=True)

# Input IDs and attention mask
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

# Forward pass to get logits
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
    logits = outputs.logits

# Shift the input IDs and logits to the right to calculate loss
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = input_ids[..., 1:].contiguous()
shift_attention_mask = attention_mask[..., 1:].contiguous()

# Flatten the tokens and apply the attention mask
loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
masked_loss = loss * shift_attention_mask.view(-1)
average_loss = masked_loss.sum() / shift_attention_mask.sum()

# Perplexity calculation
perplexity = torch.exp(average_loss)

print(f"Cross-entropy loss: {average_loss.item()}")
print(f"Perplexity: {perplexity.item()}")