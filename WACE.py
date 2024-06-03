import os
import re
import numpy as np
import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertForMaskedLM, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
import warnings

# Function to read JSON file
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Suppress warnings related to resource tracking
warnings.filterwarnings('ignore', message='resource_tracker: There appear to be .* leaked semaphore objects')

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased', attn_implementation="eager")

# Load GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

def tokenize_text(text, tokenizer, max_length=511):
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_length)
    return tokens

def normalize_distribution(probs):
    return probs / np.sum(probs)

def top_p_sampling(logits, top_p, top_k):
    logits = torch.clamp(logits, min=-1e9, max=1e9)
    sorted_logits, sorted_indices = torch.sort(logits.float(), descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits.float(), dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    if torch.any(sorted_indices_to_remove):
        top_k = min(top_k, torch.sum(~sorted_indices_to_remove).item())
    
    sorted_indices_to_remove[..., :top_k] = False
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    
    logits = torch.clamp(logits, min=-1e9, max=1e9)
    
    return torch.distributions.Categorical(logits=logits).sample()

def get_probabilities(model, tokenizer, text, top_p, top_k):
    inputs = tokenize_text(text, tokenizer)
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits[0, -1, :].float()  # Convert logits to float
        logits = top_p_sampling(logits, top_p=top_p, top_k=top_k)
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probabilities

def compute_log_probs(model, tokenizer, text, top_p, top_k):
    inputs = tokenize_text(text, tokenizer)
    with torch.no_grad():
        outputs = model(inputs, output_attentions=True)
    logits = outputs.logits[0, -1, :].float()  # Convert logits to float
    sampled_token = top_p_sampling(logits, top_p, top_k)
    log_prob = torch.log_softmax(logits, dim=-1)[sampled_token]
    attention_mask = outputs.attentions[-1].mean(dim=1).squeeze()
    return log_prob, attention_mask

def weighted_attention_sum(attention_mask, top_w):
    return attention_mask.sum() * top_w

def compute_hellinger(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

def compute_wasserstein(p, q):
    cumulative_p = np.cumsum(p)
    cumulative_q = np.cumsum(q)
    return np.sum(np.abs(cumulative_p - cumulative_q))

def compute_kl_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    kl_div = np.sum(p * np.log(p / q))
    return kl_div

def compute_bhattacharyya(p, q):
    return -np.log(np.sum(np.sqrt(p * q)))

def compute_js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (compute_kl_divergence(p, m) + compute_kl_divergence(q, m))

def compute_cross_entropy(text, top_w, top_p, top_k):
    text = text # Raw text, not tokenized
    
    bert_probs = get_probabilities(bert_model, bert_tokenizer, text, top_p, top_k)
    gpt2_probs = get_probabilities(gpt2_model, gpt2_tokenizer, text, top_p, top_k)

    log_prob_bert, attention_mask_bert = compute_log_probs(bert_model, bert_tokenizer, text, top_p, top_k)
    log_prob_gpt2, attention_mask_gpt2 = compute_log_probs(gpt2_model, gpt2_tokenizer, text, top_p, top_k)

    weighted_attention = weighted_attention_sum(attention_mask_bert, top_w) + weighted_attention_sum(attention_mask_gpt2, top_w)
    
    binoculars_score = (torch.log(torch.tensor(bert_probs).float().exp())) / (torch.log(((torch.tensor(bert_probs).float().exp() * torch.tensor(gpt2_probs).float().exp()).mean())))
    bjørnoculars_score = (torch.log(torch.tensor(bert_probs).float().exp() + weighted_attention)) / (torch.log(((torch.tensor(bert_probs).float().exp() * torch.tensor(gpt2_probs).float().exp()).mean())))
    wasserstein_dist = compute_wasserstein(bert_probs, gpt2_probs)
    hellinger_dist = compute_hellinger(bert_probs, gpt2_probs)
    kl_divergence = compute_kl_divergence(bert_probs, gpt2_probs)
    bhattacharyya_dist = compute_bhattacharyya(bert_probs, gpt2_probs)
    js_divergence = compute_js_divergence(bert_probs, gpt2_probs)
    
    return {
        'wasserstein_distance': wasserstein_dist,
        'hellinger_distance': hellinger_dist,
        'kl_divergence': kl_divergence,
        'binoculars_cross_entropy': binoculars_score,
        'bjøroculars_cross_entropy': bjørnoculars_score,
        'bhattacharyya_distance': bhattacharyya_dist,
        'js_divergence': js_divergence
    }

def run_classification(features, labels, feature_name):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    
    print(f"Feature Set: {feature_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}\n")

def main(top_w=0.2, top_p=0.95, top_k=300):
    FILE_PATH = 'data_cleaning/prompt12_merged.json'
    data = read_json(FILE_PATH)
    if 'Answers' not in data or not data['Answers']:
        print("No answers found in the JSON data.")
        return

    human_answers = [entry['human'] for entry in data['Answers']]
    #llama2_answer = [entry['llama2'] for entry in data['Answers']]
    llama3_answers = [entry['llama3'] for entry in data['Answers']]
    gpt3_answers = [entry['chatGpt3'] for entry in data['Answers']]
    gpt4_answers = [entry['chatGpt4'] for entry in data['Answers']]
    
    
    texts = human_answers + gpt4_answers
    labels = [0] * len(human_answers) + [1] * len(gpt3_answers)

    scores = []
    total_texts = len(texts)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_cross_entropy, text, top_w, top_p, top_k) for text in texts]
        for future in tqdm(as_completed(futures), total=total_texts):
            score = future.result()
            scores.append(score)

    # Extract individual score lists
    wasserstein_distances = [score['wasserstein_distance'] for score in scores]
    hellinger_distances = [score['hellinger_distance'] for score in scores]
    kl_divergences = [score['kl_divergence'] for score in scores]
    binoculars_scores = [score['binoculars_cross_entropy'] for score in scores]
    bjørnoculars_scores = [score['bjøroculars_cross_entropy'] for score in scores]
    bhattacharyya_distances = [score['bhattacharyya_distance'] for score in scores]
    js_divergences = [score['js_divergence'] for score in scores]
    

    feature_sets = {
        'Wasserstein & Hellinger': list(zip(wasserstein_distances, hellinger_distances)),
        'KL Divergence': list(zip(kl_divergences)),
        'Binoculars Cross Entropy': list(zip(binoculars_scores)),
        'Bjørnoculars Cross Entropy': list(zip(bjørnoculars_scores)),
        'Bhattacharyya & JS Divergence': list(zip(bhattacharyya_distances, js_divergences)),
        'All Features': list(zip(wasserstein_distances, hellinger_distances, kl_divergences, binoculars_scores, bjørnoculars_scores, bhattacharyya_distances, js_divergences))
    }

    for feature_name, feature_data in feature_sets.items():
        run_classification(feature_data, labels, feature_name)

if __name__ == "__main__":
    main()
