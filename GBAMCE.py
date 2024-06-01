import os
import re
import numpy as np
import json
import torch
import pickle
from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
import warnings

# Suppress warnings related to resource tracking
warnings.filterwarnings('ignore', message='resource_tracker: There appear to be .* leaked semaphore objects')

# Text cleaning function
def clean_text(text):
    text = re.sub(r'(Introduction.*?:|Body:|Conclusion.*?:)', '', text)
    text = re.sub(r'\d+\.\s.*?\(\d+ words\):', '', text)
    return text.strip()

# Function to get contents of all files in a folder
def get_files_content(folder_path):
    files_content = []
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                content = f.read()
                cleaned_content = clean_text(content)
                files_content.append(cleaned_content)
    return files_content

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Load GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

def preprocess_text(text, tokenizer):
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=511)
    return tokenizer.decode(tokens[0])

def tokenize_text(text, tokenizer, max_length=511):
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_length)
    return tokens

def normalize_distribution(probs):
    return probs / np.sum(probs)

def top_p_sampling(logits, top_p=0.5, top_k=50):
    logits = torch.clamp(logits, min=-1e9, max=1e9)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits.float(), dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    if torch.any(sorted_indices_to_remove):
        top_k = min(top_k, torch.sum(~sorted_indices_to_remove).item())
    
    sorted_indices_to_remove[..., :top_k] = False
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    
    logits = torch.clamp(logits, min=-1e9, max=1e9)
    
    return torch.distributions.Categorical(logits=logits).sample()

def get_probabilities(model, tokenizer, text, top_p=0.5, top_k=50):
    inputs = tokenize_text(text, tokenizer)
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits[0, -1, :].float()  # Convert logits to float
        logits = top_p_sampling(logits, top_p=top_p, top_k=top_k)
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probabilities

def compute_log_probs(model, tokenizer, text, top_p=0.5, top_k=50):
    inputs = tokenize_text(text, tokenizer)
    with torch.no_grad():
        outputs = model(inputs, output_attentions=True)
    logits = outputs.logits[0, -1, :].float()  # Convert logits to float
    sampled_token = top_p_sampling(logits, top_p, top_k)
    log_prob = torch.log_softmax(logits, dim=-1)[sampled_token]
    attention_mask = outputs.attentions[-1].mean(dim=1).squeeze()
    return log_prob, attention_mask

def weighted_attention_sum(attention_mask, top_w=1.0):
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

def compute_cross_entropy(text, top_w=1.0, top_p=0.95, top_k=640):
    text = preprocess_text(text, gpt2_tokenizer)
    
    bert_probs = get_probabilities(bert_model, bert_tokenizer, text, top_p, top_k)
    gpt2_probs = get_probabilities(gpt2_model, gpt2_tokenizer, text, top_p, top_k)
    
    bert_probs = normalize_distribution(bert_probs)
    gpt2_probs = normalize_distribution(gpt2_probs)

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

def sample_data(human_texts, gpt_texts, sample_size):
    total_size = len(human_texts) + len(gpt_texts)
    if sample_size > total_size:
        sample_size = total_size

    combined_texts = human_texts + gpt_texts
    combined_labels = [0] * len(human_texts) + [1] * len(gpt_texts)
    sample_indices = random.sample(range(total_size), sample_size)
    
    sampled_texts = [combined_texts[i] for i in sample_indices]
    sampled_labels = [combined_labels[i] for i in sample_indices]
    
    return sampled_texts, sampled_labels

def main(top_w=0.2, top_p=0.95, top_k=640, sample_size=None):
    human_answers_folder = 'Test-Data/essay/human'
    gpt_answers_folder = 'Test-Data/essay/gpt'
    pickle_folder = 'pickles'
    
    human_texts = get_files_content(human_answers_folder)
    gpt_texts = get_files_content(gpt_answers_folder)

    if sample_size:
        texts, labels = sample_data(human_texts, gpt_texts, sample_size)
    else:
        texts = human_texts + gpt_texts
        labels = [0] * len(human_texts) + [1] * len(gpt_texts)

    scores = []
    total_texts = len(texts)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_cross_entropy, text, top_w, top_p, top_k) for text in texts]
        for future in tqdm(as_completed(futures), total=total_texts):
            score = future.result()
            scores.append(score)

    # Flatten the dictionary scores to feature vectors
    X = [[
        score['wasserstein_distance'], score['hellinger_distance'], score['kl_divergence'],
        score['binoculars_cross_entropy'], score['bjøroculars_cross_entropy'],
        score['bhattacharyya_distance'], score['js_divergence']
    ] for score in scores]
    y = labels

    # Save the scores and labels to pickle files
    os.makedirs(pickle_folder, exist_ok=True)
    with open(os.path.join(pickle_folder, 'scores.pkl'), 'wb') as f:
        pickle.dump(X, f)
    with open(os.path.join(pickle_folder, 'labels.pkl'), 'wb') as f:
        pickle.dump(y, f)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Save the trained model to a pickle file
    with open(os.path.join(pickle_folder, 'model.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

if __name__ == "__main__":
    top_w = 0.2
    top_p = 0.95
    top_k = 640
    sample_size = 10

    main(top_w=top_w, top_p=top_p, top_k=top_k, sample_size=sample_size)
