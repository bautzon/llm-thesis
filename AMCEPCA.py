import json
import torch
from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Function to read JSON file
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

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

def compute_cross_entropy(text, top_w=1.0, top_p=0.9, top_k=100):
    text = preprocess_text(text, gpt2_tokenizer)

    log_prob_bert, attention_mask_bert = compute_log_probs(bert_model, bert_tokenizer, text, top_p, top_k)
    log_prob_gpt2, attention_mask_gpt2 = compute_log_probs(gpt2_model, gpt2_tokenizer, text, top_p, top_k)
    
    weighted_attention = weighted_attention_sum(attention_mask_bert, top_w) + weighted_attention_sum(attention_mask_gpt2, top_w)
    cross_entropy_score = cross_entropy(log_prob_bert, log_prob_gpt2) + weighted_attention
    
    return cross_entropy_score.item()

def process_entry(entry, top_w, top_p, top_k):
    human_answer = entry.get('human', '')
    llama2_answer = entry.get('llama2', '')
    llama3_answer = entry.get('llama3', '')
    gpt3_answer = entry.get('chatGpt3', '')

    score_human = compute_cross_entropy(human_answer, top_w, top_p, top_k)
    score_llama2 = compute_cross_entropy(llama2_answer, top_w, top_p, top_k)
    score_llama3 = compute_cross_entropy(llama3_answer, top_w, top_p, top_k)
    score_gpt3 = compute_cross_entropy(gpt3_answer, top_w, top_p, top_k)

    return score_human, score_llama2, score_llama3, score_gpt3

def main(top_w=1.0, top_p=0.9, top_k=100):
    FILE_PATH = 'data_cleaning/prompt2_merged.json'
    data = read_json(FILE_PATH)
    
    if 'Answers' not in data or not data['Answers']:
        print("No answers found in the JSON data.")
        return

    all_scores_human = []
    all_scores_ai = []
    labels = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_entry, entry, top_w, top_p, top_k) for entry in data['Answers']]
        for future in tqdm(as_completed(futures), total=len(futures)):
            scores_human, scores_llama2, scores_llama3, scores_gpt3 = future.result()

            all_scores_human.append(scores_human)
            labels.append(0)  # Human label
            all_scores_ai.extend([scores_llama2, scores_llama3, scores_gpt3])
            labels.extend([1, 1, 1])  # AI label for Llama2, Llama3, and GPT-3

    all_scores = all_scores_human + all_scores_ai
    X = [[score] for score in all_scores]
    y = labels

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

if __name__ == "__main__":
    # Set your parameters here
    top_w = 0.05
    top_p = 0.95
    top_k = 300

    main(top_w=top_w, top_p=top_p, top_k=top_k)
    
    
    
""" (Accuracy: 0.725
Classification Report:
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        22
           1       0.72      1.00      0.84        58

    accuracy                           0.72        80
   macro avg       0.36      0.50      0.42        80
weighted avg       0.53      0.72      0.61        80
)
"""

"""


"""