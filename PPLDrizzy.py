import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from IPython.display import HTML

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Example text
text = "This is a test sentence for visualizing perplexity."

# Encode text
encodings = tokenizer(text, return_tensors='pt')

# Calculate loss
input_ids = encodings.input_ids
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    perplexities = torch.exp(outputs.logits)

# Compute normalized perplexity for each token
logits = outputs.logits.squeeze()
token_ppls = []
for i in range(len(logits)):
    token = logits[i]
    token_id = input_ids.squeeze()[i].item()
    target = torch.tensor([token_id]).unsqueeze(0)
    with torch.no_grad():
        log_prob = torch.log_softmax(token, dim=-1)
        token_ppl = torch.exp(-log_prob.gather(1, target)).item()
        token_ppls.append(token_ppl)

# Normalize perplexities
max_ppl = max(token_ppls)
normalized_ppls = [ppl / max_ppl for ppl in token_ppls]

def generate_html(tokens, scores):
    html = "<p>"
    for token, score in zip(tokens, scores):
        color_value = int(255 * (1 - score))  # Green for low perplexity, Red for high perplexity
        html += f"<span style='background-color: rgb(255, {color_value}, {color_value}); color: black;'>{token}</span> "
    html += "</p>"
    return html

# Decode tokens
tokens = [tokenizer.decode([tok], clean_up_tokenization_spaces=False) for tok in input_ids.squeeze().tolist()]
html_output = generate_html(tokens, normalized_ppls)

# Display the HTML
display(HTML(html_output))