import json
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.spatial import distance

# Load the Word2Vec model
model_path = '/Users/bau/Library/CloudStorage/OneDrive-ITU/Thesis/llm-thesis/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Read JSON file
def read_json_file(file_path='/Users/bau/Library/CloudStorage/OneDrive-ITU/Thesis/llm-thesis/test_data.json'):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['comparisons']

# Convert text to vector using the Word2Vec model
def text_to_vector(text, model):
    words = text.split()
    word_vectors = np.array([model[word] for word in words if word in model])
    if word_vectors.size > 0:
        return np.mean(word_vectors, axis=0)
    return None

# Generate word pairs from text
def generate_word_pairs(text):
    words = text.split()
    return [(words[i], words[i + 1]) for i in range(len(words) - 1)]

# Calculate Euclidean distances for word pairs and return average
def calculate_distances_for_text(text, model):
    word_pairs = generate_word_pairs(text)
    distances = [distance.euclidean(model[word1], model[word2])
                 for word1, word2 in word_pairs if word1 in model and word2 in model]
    if distances:
        return np.mean(distances)
    return None

# Process the data
json_data = read_json_file()
text1_distances = []
text2_distances = []

for comparison in json_data:
    avg_dist1 = calculate_distances_for_text(comparison['text1'], model)
    avg_dist2 = calculate_distances_for_text(comparison['text2'], model)
    if avg_dist1 is not None:
        text1_distances.append(avg_dist1)
    if avg_dist2 is not None:
        text2_distances.append(avg_dist2)

# Plotting the data
plt.figure(figsize=(10, 5))
plt.plot(text1_distances, label='Text 1 Distances', marker='o', color='b')
plt.plot(text2_distances, label='Text 2 Distances', marker='x', color='r')
plt.title('Average Euclidean Distances for Word Pairs from Text1 vs. Text2')
plt.xlabel('Comparison Index')
plt.ylabel('Average Distance')
plt.legend()
plt.grid(True)
plt.show()
