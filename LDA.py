import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import nltk
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from GB_datahandler import read_and_clean_files

nltk.download('punkt')

# Load pre-trained word2vec model
MODEL_PATH = 'models/GoogleNews-vectors-negative300.bin'
MODEL = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

# Load the cleaned human and GPT answers
human_answers_folder = 'path/to/human/answers'
gpt_answers_folder = 'path/to/gpt/answers'
sample_size = 100  # Adjust as needed

gb_human_answers = read_and_clean_files(human_answers_folder, sample_size=sample_size)
gb_gpt_answers = read_and_clean_files(gpt_answers_folder, sample_size=sample_size)

# Combine answers and create labels
answers = gb_human_answers + gb_gpt_answers
creators = [1] * len(gb_human_answers) + [0] * len(gb_gpt_answers)

# Function to check if a word is in the model
def word_in_model(word, model):
    return word in model.key_to_index

# Initialize lists to collect data
data = []

# Process each answer
for answer_id, (text, creator) in enumerate(zip(answers, creators)):
    # Tokenize text
    words = word_tokenize(text)

    # Calculate cosine similarity, Euclidean distance, and variance
    cos_similarities = []
    euclidean_dists = []
    word_vectors = []

    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i+1]
        if word_in_model(word1, MODEL) and word_in_model(word2, MODEL):
            vec1 = MODEL[word1].reshape(1, -1)
            vec2 = MODEL[word2].reshape(1, -1)

            cos_sim = cosine_similarity(vec1, vec2)[0][0]
            euclidean_dist = euclidean_distances(vec1, vec2)[0][0]

            cos_similarities.append(cos_sim)
            euclidean_dists.append(euclidean_dist)
            word_vectors.append(vec1[0])
        else:
            cos_similarities.append(np.nan)
            euclidean_dists.append(np.nan)

    # Calculate the mean vector and variance
    if word_vectors:
        mean_vector = np.nanmean(word_vectors, axis=0)
        variances = []

        for vec in word_vectors:
            variance = np.sum((vec - mean_vector) ** 2)
            variances.append(variance)
    else:
        variances = [np.nan] * (len(words) - 1)

    # Collect data for this answer
    for i in range(len(cos_similarities)):
        data.append([answer_id, words[i], words[i + 1], cos_similarities[i], euclidean_dists[i], variances[i], creator])

# Create the DataFrame
df = pd.DataFrame(data, columns=['Answer_ID', 'Current_Word', 'Next_Word', 'Cosine_Similarity', 'Euclidean_Distance', 'Variance', 'Creator'])

# Print the initial DataFrame size
print(f"Initial DataFrame size: {df.shape}")

# Drop rows with NaN values
df = df.dropna()

# Print the size after dropping NaNs
print(f"DataFrame size after dropping NaNs: {df.shape}")

# Check the first few rows of the DataFrame
print(df.head())

# Select features and label
X = df[['Cosine_Similarity', 'Euclidean_Distance', 'Variance']]
y = df['Creator']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Perform LDA with 1 component (binary classification)
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Print the explained variance ratio
print('Explained variance ratio:', lda.explained_variance_ratio_)

# Optionally, visualize the transformed data
plt.figure(figsize=(8,6))
for label, marker, color in zip([0, 1], ('^', 's'), ('blue', 'red')):
    plt.scatter(x=X_train_lda[y_train == label, 0],
                y=np.zeros_like(X_train_lda[y_train == label, 0]),
                marker=marker,
                color=color,
                alpha=0.7,
                label=f'Class {label}')
plt.xlabel('LD1')
plt.legend()
plt.title('LDA: Linear Discriminant Analysis')
plt.show()
