import json
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.spatial import distance

# Load the Word2Vec model
#Todo! Add URL to source
model_path = '/Users/bau/Library/CloudStorage/OneDrive-ITU/Thesis/llm-thesis/GoogleNews-vectors-negative300.bin' 
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Read JSON file
def read_json_file(file_path='/Users/bau/Library/CloudStorage/OneDrive-ITU/Thesis/llm-thesis/Test-Data/combined.json'):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Answers']

def generate_word_pairs(text):
    words = text.split()
    return [(words[i], words[i + 1]) for i in range(len(words) - 1)]

def calculate_distances_for_text(text, model):
    word_pairs = generate_word_pairs(text)
    distances = [distance.euclidean(model[word1], model[word2])
                 for word1, word2 in word_pairs if word1 in model and word2 in model]
    if distances:
        return np.mean(distances)
    print(word_pairs)
    return None

# Process the data
json_data = read_json_file()
text1_distances = []
text2_distances = []

# previous_word = None
# for word in range(len(wordList) -1):
#     current_word = wordList[word +1]
#     if previous_word is not None:
#         vector_distance = calculate_distances_for_text(comparison['answer'], model)
#         print(f"Vector distance between '{current_word}' and '{previous_word}': {vector_distance}")
#     previous_word = current_word

for comparison in json_data:
    avg_dist1 = calculate_distances_for_text(comparison['prompt'], model)
    avg_dist2 = calculate_distances_for_text(comparison['answer'], model)
    if avg_dist1 is not None:
        text1_distances.append(avg_dist1)
    if avg_dist2 is not None:
        text2_distances.append(avg_dist2)

# Calculate derivatives of the distance arrays
text1_derivatives = np.gradient(text1_distances)
text2_derivatives = np.gradient(text2_distances)

# Plotting the original distance data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(text1_distances, label='Text 1 Distances', marker='o', color='b')
plt.plot(text2_distances, label='Text 2 Distances', marker='x', color='r')
plt.title('Average Euclidean Distances for Word Pairs')
plt.xlabel('Number of Answers')
plt.ylabel('Average Distance')
plt.legend()
plt.grid(True)

# Plotting the derivatives of the distances
plt.subplot(1, 2, 2)
plt.plot(text1_derivatives, label='Derivative of Text 1 Distances', marker='o', linestyle='--', color='b')
plt.plot(text2_derivatives, label='Derivative of Text 2 Distances', marker='x', linestyle='--', color='r')
plt.title('Derivatives of Distances')
plt.xlabel('Number of Tokens')
plt.ylabel('Rate of Change in Distance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
