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

human_creator = ['human'] * len(prompt1_human.distances)
ai_creator = ['ai'] * len(prompt1_llama2_student.distances)

llama3_dataFrame = pd.DataFrame({
    "distance": prompt1_llama3_student.distances.extend(prompt1_human.distances),
    "covariance": prompt1_llama3_student.covariances.extend(prompt1_human.covariances),
    "cosine variance": prompt1_llama3_student.cosine_variance_per_answer.extend(prompt1_human.cosine_variance_per_answer),
    "creator": ai_creator.extend(human_creator)
})

features = llama3_dataFrame[['distance', 'covariance', 'cosine variance']]

target = llama3_dataFrame['creator']

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(features_standardized)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

target = ['human', 'ai']

colors = ['r', 'g']

for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

fig.show()



def get_json_data():
    prompt1_llama2_student_json_data = read_json_file('Test-Data/prompt1_llama2_student.json')
    prompt1_llama3_student_json_data = read_json_file('Test-Data/combined.json')
    prompt1_chatGpt3_student_json_data = read_json_file('Test-Data/output_chatGpt_prompt1_Student.json')
    prompt1_chatGpt4_student_json_data = read_json_file('Test-Data/prompt1_gpt4_student.json')
    
    prompt2_llama2_student_json_data = read_json_file('Test-Data/prompt2_llama2_student.json')
    prompt2_chatGpt4_student_json_data = read_json_file('Test-Data/prompt2_gpt4_student.json')
    prompt2_llama3_json_data = read_json_file('Test-Data/prompt_2_ai_output.json')
    prompt2_chatGpt3_json_data = read_json_file('Test-Data/output_chatGpt_prompt2_Student.json')
    
    prompt1_llama2_student_answers = [answer['answer'] for answer in prompt1_llama2_student_json_data]
    prompt1_llama3_student_answers = [answer['answer'] for answer in prompt1_llama3_student_json_data]
    prompt1_chatGpt3_student_answers = [answer['answer'] for answer in prompt1_chatGpt3_student_json_data]
    prompt1_chatGpt4_student_answers = [answer['answer'] for answer in prompt1_chatGpt4_student_json_data]
    
    prompt2_llama2_student_answers = [answer['answer'] for answer in prompt2_llama2_student_json_data]
    prompt2_chatGpt4_student_answers = [answer['answer'] for answer in prompt2_chatGpt4_student_json_data]
    prompt2_llama3_student_answers = [answer['answer'] for answer in prompt2_llama3_json_data]
    prompt2_chatGpt3_answers = [answer['answer'] for answer in prompt2_chatGpt3_json_data]
    
    return prompt1_llama2_student_answers, prompt1_llama3_student_answers, prompt1_chatGpt3_student_answers, prompt1_chatGpt4_student_answers, prompt2_llama2_student_answers, prompt2_chatGpt4_student_answers, prompt2_llama3_student_answers, prompt2_chatGpt3_answers

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data['Answers']
    except FileNotFoundError:
        print('File not found')
        return []
    except json.JSONDecodeError:
        print(f'Invalid JSON format in {file_path}.')
        return []

def get_word_embedding(word):
    # return the embedding for the word if it exists in the model, otherwise return NaN
    return MODEL[word] if word in MODEL else np.nan
    
def calculate_euclidean_distance(embeddings):
    distances = []
    if len(embeddings) < 2:
        return np.array([])  # Not enough words to compare
    for current_word, next_word in embeddings:
        distance = dist(current_word, next_word)
        distances.append(distance)
    return distances

def get_embeddings_for_string(text):
    """
    Returns a nested list of embeddings for each word in the input text.

    Parameters:
    text (str): The input string containing words.
    model (dict): The model containing word embeddings.

    Returns:
    list: A nested list of embeddings for each word in the text.
    """
    words = text.split()
    return [get_word_embedding(word, MODEL) for word in words]

def process_pairs(embeddings):
    pairs = []

    for i in range(len(embeddings)-1):
        current_word = embeddings[i]
        next_word = embeddings[i + 1]
        if isinstance(current_word, list) and isinstance(next_word, list):
            pairs.append((current_word, next_word))
    return pairs
    
def process_triplets(embeddings):
    """
    Processes a nested list of embeddings to create lists of Previous_word, 
    Current_word, and Next_word for each word.

    Parameters:
    embeddings (list): A nested list of word embeddings.

    Returns:
    list: A list of tuples containing Previous_word, Current_word, and Next_word embeddings.
    """
    processed = []

    for i in range(len(embeddings)):
        if i == 0:
            # First word, no previous word
            continue
        elif i == len(embeddings) - 1:
            # Last word, no next word
            continue
        
        Previous_word = embeddings[i - 1]
        Current_word = embeddings[i]
        Next_word = embeddings[i + 1]
        
        processed.append((Previous_word, Current_word, Next_word))

    return processed


    #            [I][love][cats]
    #               /  | \             
    #             /    |  \ 
    #               [previous_cosine] [Current_embed] [Next_cosine] [distance_prev] [] [distance_next]
    # First word  | 
    # Second word |
    #
    #

def calculate_cosine_similarity(triplets):
    """
    Calculates the cosine similarity between the Current_word and both Previous_word and Next_word.

    Parameters:
    triplets (list): A list of tuples containing Previous_word, Current_word, and Next_word embeddings.

    Returns:
    list: A list of tuples containing the cosine similarities (prev_similarity, next_similarity).
    """
    similarities = []

    for prev, curr, next_ in triplets:
        if isinstance(prev, np.ndarray) and isinstance(curr, np.ndarray) and isinstance(next_, np.ndarray):
            prev_similarity = 1 - cosine(prev, curr)
            next_similarity = 1 - cosine(curr, next_)
        else:
            prev_similarity = np.nan
            next_similarity = np.nan
        
        similarities.append((prev_similarity, next_similarity))

    return similarities

def calculate_mean_cosine_similarity(pairs):
    """
    Calculates the mean cosine similarity between the Current_word and Next_word for each pair.

    Parameters:
    pairs (list): A list of tuples containing Current_word and Next_word embeddings.

    Returns:
    float: The mean cosine similarity.
    """
    cosine_similarities = []

    for curr, next_ in pairs:
        similarity = 1 - cosine(curr, next_)
        cosine_similarities.append(similarity)
    
    if cosine_similarities:
        mean_cosine_similarity = np.mean(cosine_similarities)
    else:
        mean_cosine_similarity = np.nan

    return mean_cosine_similarity 

def calculate_euclidean_distance(triplets):
    """
    Calculates the Euclidean distance between the Current_word and both Previous_word and Next_word.

    Parameters:
    triplets (list): A list of tuples containing Previous_word, Current_word, and Next_word embeddings.

    Returns:
    list: A list of tuples containing the Euclidean distances (prev_distance, next_distance).
    """
    distances = []

    for prev, curr, next_ in triplets:
        if isinstance(prev, list) and isinstance(curr, list) and isinstance(next_, list):
            prev_distance = euclidean(prev, curr)
            next_distance = euclidean(curr, next_)
        else:
            prev_distance = np.nan
            next_distance = np.nan
        
        distances.append((prev_distance, next_distance))

    return distances

def calculate_mean_euclidean_distance(pairs):
    """
    Calculates the mean Euclidean distance between the Current_word and Next_word for each pair.

    Parameters:
    pairs (list): A list of tuples containing Current_word and Next_word embeddings.

    Returns:
    float: The mean Euclidean distance.
    """
    euclidean_distances = []

    for curr, next_ in pairs:
        distance = euclidean(curr, next_)
        euclidean_distances.append(distance)
    
    if euclidean_distances:
        mean_euclidean_distance = np.mean(euclidean_distances)
    else:
        mean_euclidean_distance = np.nan

    return mean_euclidean_distance

def calculate_mean(numbers):
    """
    Calculates the mean of a list of numbers.

    Parameters:
    numbers (list): A list of numeric values.

    Returns:
    float: The mean of the list of numbers.
    """
    if not numbers:  # Check if the list is empty
        return 0

    mean = np.mean(numbers)
    
    return mean


# main
# prompt1_llama2_student_answers, prompt1_llama3_student_answers, prompt1_chatGpt3_student_answers, prompt1_chatGpt4_student_answers, prompt2_llama2_student_answers, prompt2_chatGpt4_student_answers, prompt2_llama3_answers, prompt2_chatGpt3_answers = get_json_data()
