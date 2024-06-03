import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration

# Load the Falcon model (for numerator)
t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
# Load the Prometheus model (for denominator)
prometheus_tokenizer = AutoTokenizer.from_pretrained("prometheus-eval/prometheus-8x7b-v2.0")
prometheus_model = AutoModelForCausalLM.from_pretrained("prometheus-eval/prometheus-8x7b-v2.0")


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def preprocess_text(text, tokenizer):
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=511)
    return tokenizer.decode(tokens[0])

def tokenize_text(text, tokenizer, max_length=511):
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_length)
    return tokens

def top_p_sampling(logits, top_p=0.5, top_k=50):
    logits = torch.clamp(logits, min=-1e9, max=1e9)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    if torch.any(sorted_indices_to_remove):
        top_k = min(top_k, torch.sum(~sorted_indices_to_remove).item())
    
    sorted_indices_to_remove[..., :top_k] = False
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    
    logits = torch.clamp(logits, min=-1e9, max=1e9)
    
    return torch.distributions.Categorical(logits=logits).sample()

def compute_log_probs(model, tokenizer, text, top_p=0.5, top_k=50):
    inputs = tokenize_text(text, tokenizer)
    with torch.no_grad():
        outputs = model(inputs, output_attentions=True)
    logits = outputs.logits[0, -1, :]

    sampled_token = top_p_sampling(logits, top_p, top_k)
    log_prob = torch.log_softmax(logits, dim=-1)[sampled_token]
    
    attention_mask = outputs.attentions[-1].mean(dim=1).squeeze()
    return log_prob, attention_mask

def weighted_attention_sum(attention_mask, top_w=1.0):
    return attention_mask.sum() * top_w

def cross_entropy(log_probs_1, log_probs_2):
    return -(log_probs_1 * log_probs_2).sum()

def compute_binoculars_score(text, top_w=1.0, top_p=0.9, top_k=100):
    text = preprocess_text(text, falcon_tokenizer)

    # Compute log probabilities and attention masks for Falcon base (numerator)
    log_prob_falcon, attention_mask_falcon = compute_log_probs(falcon_model, falcon_tokenizer, text, top_p, top_k)

    # Compute log probabilities for Prometheus model (denominator)
    log_prob_prometheus, _ = compute_log_probs(prometheus_model, prometheus_tokenizer, text, top_p, top_k)

    # Compute weighted attention sum for Falcon base
    weighted_attention_falcon = weighted_attention_sum(attention_mask_falcon, top_w)

    # Compute the numerator: log perplexity from Falcon base + weighted attention
    numerator = torch.log(log_prob_falcon.exp())

    # Compute the denominator: cross perplexity between Falcon base and Prometheus
    cross_perplexity = torch.log(((log_prob_falcon.exp() * log_prob_prometheus.exp()).mean()))
    denominator = cross_perplexity

    # Compute the Binoculars score
    binoculars_score = numerator / (denominator + weighted_attention_falcon)

    return binoculars_score.item()

def main():
    FILE_PATH = 'data_cleaning/prompt1_merged.json'
    data = read_json(FILE_PATH)
    
    if 'Answers' not in data or not data['Answers']:
        print("No answers found in the JSON data.")
        return
    
    answers = data['Answers']
    I = 1
    
    human_answer = answers[I].get('human', '')
    llama3_answer = answers[I].get('llama3', '')
    gpt3_answer = answers[I].get('chatGpt3', '')
    gpt4_answer = answers[I].get('chatGpt4', '')

    top_k = 300
    top_p = 0.95
    top_w = 0

    # Compute Binoculars scores
    score_human = compute_binoculars_score(human_answer, top_w=top_w, top_p=top_p, top_k=top_k)
    score_llama3 = compute_binoculars_score(llama3_answer, top_w=top_w, top_p=top_p, top_k=top_k)
    score_gpt3 = compute_binoculars_score(gpt3_answer, top_w=top_w, top_p=top_p, top_k=top_k)
    score_gpt4 = compute_binoculars_score(gpt4_answer, top_w=top_w, top_p=top_p, top_k=top_k)
    
    print(f"Binoculars score for human answer: {score_human}")
    print(f"Binoculars score for llama3 answer: {score_llama3}")
    print(f"Binoculars score for gpt3 answer: {score_gpt3}")
    print(f"Binoculars score for gpt4 answer: {score_gpt4}")

if __name__ == "__main__":
    main()