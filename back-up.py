"""https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277
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
import json

# Define the file path
file_path = 'Test-Data/combined.json'
# Read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize the list to hold all human answers
human_answers = []
synthetic_answers=[]

# Loop through each entry in the "Answers" list
for entry in data['Answers']:
    if entry['creator'] == 'human':
        # Split the answer into words and append to human_answers
        human_answer = entry['answer'].split()
        human_answers.append(human_answer)
    if entry['creator'] == 'ai':
        synthetic_answer = entry['answer'].split()
        synthetic_answers.append(synthetic_answer)


def get_top_p_predictions(model, tokenizer, prompt, top_p=1.0, top_k=100):
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

    predictions = []
    for prob, idx in zip(top_p_probs, top_p_indices):
        word = tokenizer.decode(idx.item()).strip()
        predictions.append((word, prob.item()))

    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:top_k]

def get_word_rank(predictions, target_word):
    for rank, (word, prob) in enumerate(predictions, 1):
        if word == target_word:
            return rank, prob
    return None, None

def plot_comparison(word_list1, word_list2, model, tokenizer, top_p=1.0, top_k=100):
    def get_ranks_and_probs(word_list):
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
    
    ranks1, probs1 = get_ranks_and_probs(word_list1)
    ranks2, probs2 = get_ranks_and_probs(word_list2)
    
    x_labels = [f'Word {i}' for i in range(1, max(len(word_list1), len(word_list2)))]

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(ranks1) + 1), ranks1, label='Rank of Next Word - Human', color='blue')
    plt.plot(range(1, len(ranks2) + 1), ranks2, label='Rank of Next Word - Synthetic', color='red')
    #plt.xticks(range(1, max(len(ranks1), len(ranks2)) + 1), x_labels, rotation=45)
    plt.xlabel('Word in the List')
    plt.ylabel('Rank')
    plt.title('Comparison of Word Prediction Ranks')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(probs1) + 1), probs1, label='Probability of Next Word - Human', color='blue')
    plt.plot(range(1, len(probs2) + 1), probs2, label='Probability of Next Word - Synthetic', color='red')
    #plt.xticks(range(1, max(len(probs1), len(probs2)) + 1), x_labels, rotation=45)
    plt.xlabel('Word in the List')
    plt.ylabel('Probability')
    plt.title('Comparison of Word Prediction Probabilities')
    plt.legend()
    plt.grid(True)
    plt.show()

# Load the pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Example usage
word_list1 = ["The", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
word_list2 = ["A", "fast", "dark", "wolf", "leaped", "across", "a", "sleepy", "cat"]
#plot_comparison(word_list1, word_list2, model, tokenizer)

# Plot the comparison
plot_comparison(human_answers[0], synthetic_answers[0], model, tokenizer)
first_entry_hum = human_answers[0]
first_entry_syn = synthetic_answers[0]
plot_comparison(first_entry_hum, first_entry_syn, model, tokenizer)


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json

# Define the file path
file_path = 'Test-Data/combined.json'

# Read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize the list to hold all human answers
human_answers = []
synthetic_answers = []

# Loop through each entry in the "Answers" list
for entry in data['Answers']:
    if entry['creator'] == 'human':
        # Split the answer into words and append to human_answers
        human_answer = entry['answer'].split()
        human_answers.append(human_answer)
    if entry['creator'] == 'ai':
        synthetic_answer = entry['answer'].split()
        synthetic_answers.append(synthetic_answer)

def get_top_p_predictions(model, tokenizer, prompt, top_p=1.0, top_k=100):
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

    predictions = []
    for prob, idx in zip(top_p_probs, top_p_indices):
        word = tokenizer.decode(idx.item()).strip()
        predictions.append((word, prob.item()))

    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:top_k]

def get_word_rank(predictions, target_word):
    for rank, (word, prob) in enumerate(predictions, 1):
        if word == target_word:
            return rank, prob
    return None, None

def plot_comparison(word_list1, word_list2, model, tokenizer, top_p=1.0, top_k=1000):
    def get_ranks_and_probs(word_list):
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
    
    ranks1, probs1 = get_ranks_and_probs(word_list1)
    ranks2, probs2 = get_ranks_and_probs(word_list2)
    
    plt.figure(figsize=(12, 6))
    plt.hist(ranks1, bins=2, alpha=0.5, label='Rank of Next Word - Human', color='b')
    plt.hist(ranks2, bins=2, alpha=0.5, label='Rank of Next Word - Synthetic', color='r')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Histogram of Word Prediction Ranks')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(12, 6))
    plt.hist(probs1, bins=2, alpha=0.5, label='Probability of Next Word - Human', color='b')
    plt.hist(probs2, bins=2, alpha=0.5, label='Probability of Next Word - Synthetic', color='r')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Histogram of Word Prediction Probabilities')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()

# Load the pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Example usage
first_entry_hum = human_answers[0]
first_entry_syn = synthetic_answers[0]
plot_comparison(first_entry_hum, first_entry_syn, model, tokenizer)
----
import os
import re
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from math import dist
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine, euclidean
from DistanceMatrix import get_openqa_calculations, get_eli5_calculations, get_prompt1_calculations, get_prompt2_calculations, CalculationsObject
import pandas as pd


MODEL = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
# MODEL = KeyedVectors.load_word2vec_format("models/crawl-300d-2M.vec")
# MODEL = KeyedVectors.load_word2vec_format("models/wiki-news-300d-1M.vec")


prompt1_human, prompt1_llama2_student, prompt1_llama3_student, prompt1_gpt3_student, prompt1_gpt3_plain, prompt1_gpt3_humanlike, prompt1_gpt4_student = get_prompt1_calculations()
prompt2_human, prompt2_llama2_student, prompt2_llama3_student, prompt2_gpt3_student, prompt2_gpt3_plain, prompt2_gpt3_humanlike, prompt2_gpt4_student = get_prompt2_calculations()
eli5_human, eli5_llama2, eli5_llama3, eli5_chatGpt3 = get_eli5_calculations()
openqa_human, openqa_chatGpt3 = get_openqa_calculations()

human_creator = ['human'] * len(eli5_human.distances)
ai_creator = ['ai'] * len(eli5_chatGpt3.distances)

llama3_dataFrame = pd.DataFrame({
    "distance": eli5_chatGpt3.distances + eli5_human.distances,
    "covariance": eli5_chatGpt3.covariances + eli5_human.covariances,
    "cosine variance": eli5_chatGpt3.cosine_variance_per_answer + eli5_human.cosine_variance_per_answer,
    "creator": ai_creator + human_creator
})

features = llama3_dataFrame[['distance', 'covariance', 'cosine variance']]

target = llama3_dataFrame['creator']

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(features_standardized)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

finalDf = pd.concat([principalDf, target], axis = 1)

fig = plt.figure(2, figsize = (8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)

targets = ['human', 'ai']

colors = ['blue', 'g']

for target, color in zip(targets, colors):
    indicesToKeep = finalDf['creator'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

plt.show()