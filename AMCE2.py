import torch
from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Load GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

def preprocess_text(text):
    return text.strip()

def tokenize_text(text, tokenizer):
    return tokenizer.encode(text, return_tensors='pt')

def top_p_sampling(logits, top_p=0.5, top_k=50):
    logits = torch.clamp(logits, min=-1e9, max=1e9)  # Clamp logits to avoid NaN
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    if torch.any(sorted_indices_to_remove):
        top_k = min(top_k, torch.sum(~sorted_indices_to_remove).item())
    
    # Ensure at least one token is kept
    sorted_indices_to_remove[..., :top_k] = False
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    
    logits = torch.clamp(logits, min=-1e9, max=1e9)  # Clamp logits again to avoid NaN
    
    return torch.distributions.Categorical(logits=logits).sample()

def compute_log_probs(model, tokenizer, text, top_p=0.5, top_k=50):
    inputs = tokenize_text(text, tokenizer)
    with torch.no_grad():
        outputs = model(inputs, output_attentions=True)
    logits = outputs.logits[0, -1, :]

    print(f"Logits before sampling: {logits}")

    # Check for NaN values in logits
    if torch.isnan(logits).any():
        print("NaN detected in logits before sampling.")

    sampled_token = top_p_sampling(logits, top_p, top_k)
    log_prob = torch.log_softmax(logits, dim=-1)[sampled_token]

    # Check for NaN values in log probabilities
    if torch.isnan(log_prob).any():
        print("NaN detected in log probabilities.")
    
    attention_mask = outputs.attentions[-1].mean(dim=1).squeeze()  # Average attention masks
    return log_prob, attention_mask

def weighted_attention_sum(attention_mask, top_w=1.0):
    return attention_mask.sum() * top_w

def cross_entropy(log_probs_1, log_probs_2):
    return -(log_probs_1 * log_probs_2).sum()

def compute_cross_entropy_for_list(sample_list, top_w=1.0, top_p=0.9, top_k=100):
    results = []
    
    for text in sample_list:
        text = preprocess_text(text)
        
        log_prob_bert, attention_mask_bert = compute_log_probs(bert_model, bert_tokenizer, text, top_p, top_k)
        log_prob_gpt2, attention_mask_gpt2 = compute_log_probs(gpt2_model, gpt2_tokenizer, text, top_p, top_k)
        
        weighted_attention = weighted_attention_sum(attention_mask_bert, top_w) + weighted_attention_sum(attention_mask_gpt2, top_w)
        
        cross_entropy_score = cross_entropy(log_prob_bert, log_prob_gpt2) + weighted_attention
        
        results.append(cross_entropy_score.item())
    
    return results

# Example usage
sample1 = ["sample 1 here  ", " another sample here"]
sample2 = ["sample 3 here  ", " a 4 sample here"]

scores_sample1 = compute_cross_entropy_for_list(sample1, top_w=0.5)
scores_sample2 = compute_cross_entropy_for_list(sample2, top_w=0.5)

print(f"Cross-entropy scores for sample1: {scores_sample1}")
print(f"Cross-entropy scores for sample2: {scores_sample2}")