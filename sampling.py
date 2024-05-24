"""
References:
https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277
https://huggingface.co/blog/how-to-generate
https://colab.research.google.com/drive/1yeLM1LoaEqTAS6D_Op9_L2pLA06uUGW1#scrollTo=pfPQuW-2u-Ps
https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/02_how_to_generate.ipynb#scrollTo=HhLKyfdbsjXc
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import pickle

# Define the file path
file_path = 'Test-Data/combined.json'

# Define paths for pickle files
human_ranks_path = 'human_ranks.pkl'
synthetic_ranks_path = 'synthetic_ranks.pkl'
human_probs_path = 'human_probs.pkl'
synthetic_probs_path = 'synthetic_probs.pkl'

# Read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize the list to hold all human and synthetic answers
human_answers = []
synthetic_answers = []

# Loop through each entry in the "Answers" list
for entry in data['Answers']:
    if entry['creator'] == 'human':
        # Split the answer into words and append to human_answers
        human_answer = entry['answer'].split()
        human_answers.append(human_answer)
    elif entry['creator'] == 'ai':  # Changed to 'elif' to avoid checking twice
        synthetic_answer = entry['answer'].split()
        synthetic_answers.append(synthetic_answer)

def get_top_p_predictions(model, tokenizer, prompt, top_p=0.96, top_k=100):
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        next_token_logits = outputs.logits[:, -1, :]

    next_token_probs = F.softmax(next_token_logits, dim=-1).squeeze()

    sorted_probs, sorted_indices = torch.sort(next_token_probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    top_p_index = (cumulative_probs > top_p).nonzero(as_tuple=True)
    if len(top_p_index[0]) > 0:
        top_p_index = top_p_index[0][0]
        top_p_probs = sorted_probs[:top_p_index + 1]
        top_p_indices = sorted_indices[:top_p_index + 1]
    else:
        top_p_probs = sorted_probs
        top_p_indices = sorted_indices

    predictions = [(tokenizer.decode(idx.item()).strip(), prob.item()) for prob, idx in zip(top_p_probs, top_p_indices)]

    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:top_k]

def get_word_rank(predictions, target_word):
    for rank, (word, prob) in enumerate(predictions, 1):
        if word == target_word:
            return rank, prob
    return None, None

def collect_ranks_and_probs(word_list, model, tokenizer, top_p=0.96, top_k=100):
    ranks = []
    probs = []
    for i in range(len(word_list) - 1):
        prompt = ' '.join(word_list[:i+1])
        next_word = word_list[i+1]
        predictions = get_top_p_predictions(model, tokenizer, prompt, top_p, top_k)
        rank, prob = get_word_rank(predictions, next_word)
        if rank is not None:
            ranks.append(rank)
            probs.append(prob)
        else:
            ranks.append(top_k + 1)  # If not in top-p, assign a rank larger than top_k
            probs.append(0)
    return ranks, probs

def count_rank_occurrences_in_range(ranks, start, end):
    return sum(1 for rank in ranks if start < rank <= end)

def count_all_rank_occurrences(ranks, increment=10):
    rank_counts = {}
    for i in range(0, 100, increment):
        start = i
        end = i + increment
        rank_counts[f'{start + 1}-{end}'] = count_rank_occurrences_in_range(ranks, start, end)
    return rank_counts

def plot_histograms(human_rank_counts, synthetic_rank_counts, human_probs, synthetic_probs):
    labels = list(human_rank_counts.keys())
    human_counts = list(human_rank_counts.values())
    synthetic_counts = list(synthetic_rank_counts.values())

    x = range(len(labels))

    plt.figure(figsize=(12, 6))
    plt.bar(x, human_counts, width=0.4, label='Human', color='blue', align='center')
    plt.bar(x, synthetic_counts, width=0.4, label='Synthetic', color='red', align='edge')
    plt.xlabel('Rank Range')
    plt.ylabel('Count')
    plt.title('Histogram of Word Prediction Rank Ranges')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.hist(human_probs, bins=10, alpha=0.5, label='Human', color='blue')
    plt.hist(synthetic_probs, bins=10, alpha=0.5, label='Synthetic', color='red')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title('Histogram of Word Prediction Probabilities')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_to_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Load the pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Check if pickled data exists
try:
    human_ranks = load_from_pickle(human_ranks_path)
    synthetic_ranks = load_from_pickle(synthetic_ranks_path)
    human_probs = load_from_pickle(human_probs_path)
    synthetic_probs = load_from_pickle(synthetic_probs_path)
    print("Loaded ranks and probabilities from pickle files.")
except (FileNotFoundError, EOFError):
    # Collect ranks and probabilities for human and synthetic answers
    first_entry_hum = human_answers[0]
    first_entry_syn = synthetic_answers[0]

    human_ranks, human_probs = collect_ranks_and_probs(first_entry_hum, model, tokenizer)
    synthetic_ranks, synthetic_probs = collect_ranks_and_probs(first_entry_syn, model, tokenizer)

    # Save ranks and probabilities to pickle files
    save_to_pickle(human_ranks, human_ranks_path)
    save_to_pickle(synthetic_ranks, synthetic_ranks_path)
    save_to_pickle(human_probs, human_probs_path)
    save_to_pickle(synthetic_probs, synthetic_probs_path)
    print("Computed and saved ranks and probabilities to pickle files.")


# Count rank occurrences in increments of 10
human_rank_counts = count_all_rank_occurrences(human_ranks)
synthetic_rank_counts = count_all_rank_occurrences(synthetic_ranks)

# Plot histograms
plot_histograms(human_rank_counts, synthetic_rank_counts, human_probs, synthetic_probs)