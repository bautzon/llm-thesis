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
import os

# Define global parameters
FILE_PATH = 'Test-Data/combined.json'
MODEL_NAME = 'gpt2'
TOP_K = 100
TOP_P = 0.5

# Define paths for pickle files
PICKLE_DIR = 'pickles'
HUMAN_RANKS_PATH = os.path.join(PICKLE_DIR, 'human_ranks.pkl')
SYNTHETIC_RANKS_PATH = os.path.join(PICKLE_DIR, 'synthetic_ranks.pkl')
HUMAN_PROBS_PATH = os.path.join(PICKLE_DIR, 'human_probs.pkl')
SYNTHETIC_PROBS_PATH = os.path.join(PICKLE_DIR, 'synthetic_probs.pkl')
HUMAN_PERPLEXITY_PATH = os.path.join(PICKLE_DIR, 'human_perplexities.pkl')
SYNTHETIC_PERPLEXITY_PATH = os.path.join(PICKLE_DIR, 'synthetic_perplexities.pkl')

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_to_pickle(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
def get_top_p_predictions(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        next_token_logits = outputs.logits[:, -1, :]

    next_token_probs = F.softmax(next_token_logits, dim=-1).squeeze()

    sorted_probs, sorted_indices = torch.sort(next_token_probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    top_p_index = (cumulative_probs > TOP_P).nonzero(as_tuple=True)
    if len(top_p_index[0]) > 0:
        top_p_index = top_p_index[0][0]
        top_p_probs = sorted_probs[:top_p_index + 1]
        top_p_indices = sorted_indices[:top_p_index + 1]
    else:
        top_p_probs = sorted_probs
        top_p_indices = sorted_indices

    predictions = [(tokenizer.decode(idx.item()).strip(), prob.item()) for prob, idx in zip(top_p_probs, top_p_indices)]

    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:TOP_K]

def get_word_rank(predictions, target_word):
    for rank, (word, prob) in enumerate(predictions, 1):
        if word == target_word:
            return rank, prob
    return None, None

def collect_ranks_and_probs(word_lists, model, tokenizer):
    all_ranks = []
    all_probs = []
    for word_list in word_lists:
        ranks, probs = [], []
        for i in range(len(word_list) - 1):
            prompt = ' '.join(word_list[:i+1])
            next_word = word_list[i+1]
            predictions = get_top_p_predictions(model, tokenizer, prompt)
            rank, prob = get_word_rank(predictions, next_word)
            if rank is not None:
                ranks.append(rank)
                probs.append(prob)
            else:
                ranks.append(TOP_K + 1)  # If not in top-p, assign a rank larger than top_k
                probs.append(0)
        all_ranks.extend(ranks)
        all_probs.extend(probs)
    return all_ranks, all_probs

def count_rank_occurrences_in_range(ranks, start, end):
    return sum(1 for rank in ranks if start < rank <= end)

def count_all_rank_occurrences(ranks, increment=10):
    rank_counts = {}
    for i in range(0, 100, increment):
        start = i
        end = i + increment
        rank_counts[f'{start + 1}-{end}'] = count_rank_occurrences_in_range(ranks, start, end)
    return rank_counts

def plot_histograms(human_rank_counts, synthetic_rank_counts, human_probs, synthetic_probs, human_perplexities, synthetic_perplexities):
    labels = list(human_rank_counts.keys())
    human_counts = list(human_rank_counts.values())
    synthetic_counts = list(synthetic_rank_counts.values())

    x = range(len(labels))
    plt.figure(figsize=(18, 18))
    
    # Plot the RANKS
    plt.subplot(3, 1, 1)
    plt.bar(x, human_counts, width=0.4, alpha=0.5, label='Human', color='blue', align='center')
    plt.bar(x, synthetic_counts, width=0.4, alpha=0.5, label='Synthetic', color='red', align='edge')
    plt.xlabel('Rank Range')
    plt.ylabel('Count')
    plt.title('Histogram of Word Prediction Rank Ranges')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Plot the PROBS
    plt.subplot(3, 1, 2)
    plt.hist(human_probs, bins=10, alpha=0.5, label='Human', color='blue')
    plt.hist(synthetic_probs, bins=10, alpha=0.5, label='Synthetic', color='red')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title('Histogram of Word Prediction Probabilities')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Plot the PERPLEXITIES
    plt.subplot(3, 1, 3)
    plt.hist(human_perplexities, bins=10, alpha=0.5, label='Human', color='blue')
    plt.hist(synthetic_perplexities, bins=10, alpha=0.5, label='Synthetic', color='red')
    plt.xlabel('Perplexity')
    plt.ylabel('Count')
    plt.title('Histogram of Perplexities')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()
    
def calculate_perplexity(prompt):
    try:
        # Load pre-trained model (weights)
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.eval()

        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Encode text
        encodings = tokenizer(prompt, return_tensors='pt')

        # Calculate loss
        input_ids = encodings.input_ids
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)

        return perplexity.item()
    except Exception as e:
        print(f"Error calculating perplexity for prompt: {prompt}")
        print(e)
        return None

def main():
    data = read_json(FILE_PATH)

    # Initialize the list to hold all human and synthetic answers
    human_answers = []
    synthetic_answers = []

    # Loop through each entry in the "Answers" list
    for entry in data['Answers']:
        if entry['creator'] == 'human':
            # Split the answer into words and append to human_answers
            human_answer = entry['answer'].split()
            human_answers.append(human_answer)
        elif entry['creator'] == 'ai':
            synthetic_answer = entry['answer'].split()
            synthetic_answers.append(synthetic_answer)

    # Load the pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    # Check if pickled data exists
    try:
        human_ranks = load_from_pickle(HUMAN_RANKS_PATH)
        synthetic_ranks = load_from_pickle(SYNTHETIC_RANKS_PATH)
        human_probs = load_from_pickle(HUMAN_PROBS_PATH)
        synthetic_probs = load_from_pickle(SYNTHETIC_PROBS_PATH)
        human_perplexities = load_from_pickle(HUMAN_PERPLEXITY_PATH)
        synthetic_perplexities = load_from_pickle(SYNTHETIC_PERPLEXITY_PATH)
        print('Loaded ranks, probabilities, and perplexities from pickle files.')

    except (FileNotFoundError, EOFError):
        # Collect ranks and probabilities for all human and synthetic answers
        human_ranks, human_probs = collect_ranks_and_probs(human_answers, model, tokenizer)
        synthetic_ranks, synthetic_probs = collect_ranks_and_probs(synthetic_answers, model, tokenizer)
        
        # Calculate perplexities for all human and synthetic answers
        human_prompts = [' '.join(answer) for answer in human_answers]
        synthetic_prompts = [' '.join(answer) for answer in synthetic_answers]
        human_perplexities = [calculate_perplexity(prompt) for prompt in human_prompts if calculate_perplexity(prompt) is not None]
        synthetic_perplexities = [calculate_perplexity(prompt) for prompt in synthetic_prompts if calculate_perplexity(prompt) is not None]

    # Save ranks, probabilities, and perplexities to pickle files
    save_to_pickle(human_ranks, HUMAN_RANKS_PATH)
    save_to_pickle(synthetic_ranks, SYNTHETIC_RANKS_PATH)
    save_to_pickle(human_probs, HUMAN_PROBS_PATH)
    save_to_pickle(synthetic_probs, SYNTHETIC_PROBS_PATH)
    save_to_pickle(human_perplexities, HUMAN_PERPLEXITY_PATH)
    save_to_pickle(synthetic_perplexities, SYNTHETIC_PERPLEXITY_PATH)
    
    print("Computed and saved ranks, probabilities, and perplexities to pickle files.")

    # Count rank occurrences in increments of 10
    human_rank_counts = count_all_rank_occurrences(human_ranks)
    synthetic_rank_counts = count_all_rank_occurrences(synthetic_ranks)

    # Plot histograms
    plot_histograms(human_rank_counts, synthetic_rank_counts, human_probs, synthetic_probs, human_perplexities, synthetic_perplexities)

if __name__ == "__main__":
    main()