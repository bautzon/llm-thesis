import torch

def top_p_logits(logits, p):
    """
    Filters logits using top-p (nucleus) sampling
    """
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Determine the cut-off point
    sorted_indices_to_keep = cumulative_probs <= p
    sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
    sorted_indices_to_keep[..., 0] = 1
    
    # Create a mask
    indices_to_remove = sorted_indices[~sorted_indices_to_keep]
    
    # Set logits that are not in the top-p to a very large negative value
    logits[indices_to_remove] = -float('Inf')
    return logits

# Example usage
def check_probabilities_with_top_p(self, in_text, topp=0.9):
    # Process input
    token_ids = self.enc(in_text, return_tensors='pt').data['input_ids'][0]
    token_ids = torch.concat([self.start_token, token_ids])
    
    # Forward through the model
    output = self.model(token_ids.to(self.device))
    all_logits = output.logits[:-1].detach().squeeze()
    
    # Apply top-p sampling
    top_p_filtered_logits = top_p_logits(all_logits, p=topp)
    
    # Convert logits to probabilities
    all_probs = torch.softmax(top_p_filtered_logits, dim=-1)
    
    y = token_ids[1:]
    
    # Sort the predictions for each timestep
    sorted_preds = torch.argsort(all_probs, dim=1, descending=True).cpu()
    
    # [(pos, prob), ...]
    real_topk_pos = list(
        [int(np.where(sorted_preds[i] == y[i].item())[0][0])
            for i in range(y.shape[0])])
    real_topk_probs = all_probs[np.arange(
        0, y.shape[0], 1), y].data.cpu().numpy().tolist()
    real_topk_probs = list(map(lambda x: round(x, 5), real_topk_probs))

    real_topk = list(zip(real_topk_pos, real_topk_probs))
    # [str, str, ...]
    bpe_strings = self.enc.convert_ids_to_tokens(token_ids[:])

    bpe_strings = [self.postprocess(s) for s in bpe_strings]

    topk_prob_values, topk_prob_inds = torch.topk(all_probs, k=40, dim=1)

    pred_topk = [list(zip(self.enc.convert_ids_to_tokens(topk_prob_inds[i]),
                            topk_prob_values[i].data.cpu().numpy().tolist()
                            )) for i in range(y.shape[0])]
    pred_topk = [[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk]

    payload = {'bpe_strings': bpe_strings,
                'real_topk': real_topk,
                'pred_topk': pred_topk}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return payload
"""Above was non working methods below is an example"""
import torch
from transformers import GPT2Tokenizer, GPT2Model

# Example setup (replace with your actual model and tokenizer)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()  # Set the model to evaluation mode

# Define the input text
input_text = "yo moms a whore?"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors='pt')

# Forward pass to get the outputs
with torch.no_grad():  # Disable gradient calculation for inference
    outputs = model(**inputs)

# Extract the logits
logits = outputs.last_hidden_state

# Print the shape and a sample of the logits
print("Shape of logits:", logits.shape)
print("Sample of logits:", logits[0, 0, :10])  # Print the first 10 logits for the first token



def top_p_logits(logits, p):
    """
    Filters logits using top-p (nucleus) sampling
    """
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Determine the cut-off point
    sorted_indices_to_keep = cumulative_probs <= p
    sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
    sorted_indices_to_keep[..., 0] = 1
    
    # Create a mask
    indices_to_remove = sorted_indices[~sorted_indices_to_keep]
    
    # Set logits that are not in the top-p to a very large negative value
    logits[indices_to_remove] = -float('Inf')
    return logits

# Apply top-p sampling to the logits
filtered_logits = top_p_logits(logits[0, 0], p=0.9)

# Convert filtered logits to probabilities
probs = torch.softmax(filtered_logits, dim=-1)

# Print the probabilities
print("Probabilities after top-p sampling:", probs)
"""
Nucleus sampling is an idea to correct for beam search repetition,
Basically chopping the distribution truncating af p percent probability mass,
re normalize and sample from the distribution left.

Relatively easy to implement

paper by Ari Holtzman et al. sees a high perplexity score, 
higher than top k=40 sampling.add()but not higher than Stochastic beam search b= 16


"""