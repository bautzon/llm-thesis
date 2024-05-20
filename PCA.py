import os
import re
import numpy as np
import random
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
random.seed(42)

def get_files_content(folder_path):
    files_content = []
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                files_content.append([line.strip() for line in f])
    return files_content

def clean_text(text):
    # Remove instructions
    text = re.sub(r'(Introduction.*?:|Body:|Conclusion.*?:)', '', text)
    # Remove section numbers and titles
    text = re.sub(r'\d+\.\s.*?\(\d+ words\):', '', text)
    return text.strip()

def read_and_clean_files(folder_path, sample_size=None):
    cleaned_data = []
    files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
    if sample_size:
        files = random.sample(files, sample_size)
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as f:
            content_list = [line.strip() for line in f]
            cleaned_content = [clean_text(text) for text in content_list if text]
            cleaned_data.append(''.join(cleaned_content))
    return cleaned_data

def get_word_embeddings(text, model):
    words = text.split()
    embeddings = [model[word] for word in words if word in model]
    return embeddings

def calculate_euclidean_distance(embeddings):
    if len(embeddings) < 2:
        return np.array([])  # Not enough words to compare
    distance_matrix = euclidean_distances(embeddings)
    return distance_matrix

def calculate_cosine_similarity(embeddings):
    if len(embeddings) < 2:
        return np.array([])  # Not enough words to compare
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def plot_heatmaps(euclidean_matrix, cosine_matrix, title):
    if len(euclidean_matrix) == 0 or len(cosine_matrix) == 0:
        print(f"{title}: Not enough words to plot.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot Euclidean distance matrix
    axes[0].imshow(euclidean_matrix, interpolation='nearest', cmap='viridis')
    axes[0].set_title(f'Euclidean Distance - {title}')
    axes[0].set_xlabel('Words')
    axes[0].set_ylabel('Words')
    fig.colorbar(axes[0].images[0], ax=axes[0])

    # Plot Cosine similarity matrix
    axes[1].imshow(cosine_matrix, interpolation='nearest', cmap='viridis')
    axes[1].set_title(f'Cosine Similarity - {title}')
    axes[1].set_xlabel('Words')
    axes[1].set_ylabel('Words')
    fig.colorbar(axes[1].images[0], ax=axes[1])
    
    plt.show()

def perform_pca(embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def plot_pca_and_distances(pca_embeddings, euclidean_matrix, cosine_matrix, title):
    fig, ax = plt.subplots(1, 3, figsize=(30, 8))
    
    # Plot PCA
    ax[0].scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], edgecolor='k', s=50)
    ax[0].set_title(f'PCA of {title}')
    ax[0].set_xlabel('Principal Component 1')
    ax[0].set_ylabel('Principal Component 2')
    
    # Plot Euclidean distance matrix
    cax1 = ax[1].imshow(euclidean_matrix, interpolation='nearest', cmap='viridis')
    ax[1].set_title(f'Euclidean Distance - {title}')
    ax[1].set_xlabel('Words')
    ax[1].set_ylabel('Words')
    fig.colorbar(cax1, ax=ax[1])

    # Plot Cosine similarity matrix
    cax2 = ax[2].imshow(cosine_matrix, interpolation='nearest', cmap='viridis')
    ax[2].set_title(f'Cosine Similarity - {title}')
    ax[2].set_xlabel('Words')
    ax[2].set_ylabel('Words')
    fig.colorbar(cax2, ax=ax[2])
    
    plt.show()

# Example usage
prompts_folder = 'Test-Data/essay/prompts'
human_answers_folder = 'Test-Data/essay/human'
gpt_answers_folder = 'Test-Data/essay/gpt'

# Set sample size for downsampling
sample_size = 10

gb_prompts = read_and_clean_files(prompts_folder, sample_size=sample_size)
gb_human_answers = read_and_clean_files(human_answers_folder, sample_size=sample_size)
gb_gpt_answers = read_and_clean_files(gpt_answers_folder, sample_size=sample_size)

# Load pre-trained Word2Vec model from the specified path
model_path = 'models/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Compute word embeddings
gb_prompts_embeddings = [get_word_embeddings(text, model) for text in gb_prompts[:10]]
gb_human_answers_embeddings = [get_word_embeddings(text, model) for text in gb_human_answers[:10]]
gb_gpt_answers_embeddings = [get_word_embeddings(text, model) for text in gb_gpt_answers[:10]]

# Concatenate all embeddings for standardization
all_embeddings = np.vstack(gb_prompts_embeddings + gb_human_answers_embeddings + gb_gpt_answers_embeddings)

# Standardize the embeddings
scaler = StandardScaler()
all_embeddings_standardized = scaler.fit_transform(all_embeddings)

# Function to split standardized embeddings back into original structure
def split_embeddings(original_embeddings, standardized_embeddings):
    split_standardized_embeddings = []
    start = 0
    for embeddings in original_embeddings:
        end = start + len(embeddings)
        split_standardized_embeddings.append(standardized_embeddings[start:end])
        start = end
    return split_standardized_embeddings

# Split the standardized embeddings back into their respective lists
gb_prompts_embeddings_standardized = split_embeddings(gb_prompts_embeddings, all_embeddings_standardized)
gb_human_answers_embeddings_standardized = split_embeddings(gb_human_answers_embeddings, all_embeddings_standardized)
gb_gpt_answers_embeddings_standardized = split_embeddings(gb_gpt_answers_embeddings, all_embeddings_standardized)

# Calculate Euclidean distance and cosine similarity matrices for the first 10 elements
gb_prompts_euclidean = [calculate_euclidean_distance(embeddings) for embeddings in gb_prompts_embeddings_standardized]
gb_human_answers_euclidean = [calculate_euclidean_distance(embeddings) for embeddings in gb_human_answers_embeddings_standardized]
gb_gpt_answers_euclidean = [calculate_euclidean_distance(embeddings) for embeddings in gb_gpt_answers_embeddings_standardized]

gb_prompts_cosine = [calculate_cosine_similarity(embeddings) for embeddings in gb_prompts_embeddings_standardized]
gb_human_answers_cosine = [calculate_cosine_similarity(embeddings) for embeddings in gb_human_answers_embeddings_standardized]
gb_gpt_answers_cosine = [calculate_cosine_similarity(embeddings) for embeddings in gb_gpt_answers_embeddings_standardized]

# Perform PCA on standardized embeddings
pca_prompts = [perform_pca(embeddings) for embeddings in gb_prompts_embeddings_standardized]
pca_human_answers = [perform_pca(embeddings) for embeddings in gb_human_answers_embeddings_standardized]
pca_gpt_answers = [perform_pca(embeddings) for embeddings in gb_gpt_answers_embeddings_standardized]

# Plot PCA and distances for the first 10 elements
for i in range(10):
    plot_pca_and_distances(pca_prompts[i], gb_prompts_euclidean[i], gb_prompts_cosine[i], f"Prompt {i+1}")
    plot_pca_and_distances(pca_human_answers[i], gb_human_answers_euclidean[i], gb_human_answers_cosine[i], f"Human Answer {i+1}")
    plot_pca_and_distances(pca_gpt_answers[i], gb_gpt_answers_euclidean[i], gb_gpt_answers_cosine[i], f"GPT Answer {i+1}")
