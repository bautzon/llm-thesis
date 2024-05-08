import json
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.spatial import distance

# Load the Word2Vec model
#Todo! Add URL to source
model_path = 'GoogleNews-vectors-negative300.bin' 
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Read JSON file
def read_json_file(file_path='Test-Data/combined.json'):
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
    # print(word_pairs)
    return None

# Process the data
json_data = read_json_file()
text1_distances = []
text2_distances = []

def get_word_embedding(token, model):
    try:
        embedding = model[token]
        return embedding
    except KeyError:
        # If the token is not found in the model's vocabulary, return None or a random vector
        return None
    
def calculate_distance(current_word, previous_word):
    if previous_word in model and current_word in model:
        return distance.euclidean(model[previous_word], model[current_word])
    else:
        return None

def calculate_vector_norms(words, model):
    """Calculate norms of word embeddings present in the model."""
    norms = [np.linalg.norm(model[word]) for word in words if word in model]
    return norms


    

human_vector_list = []
human_avg_len_list = []
synthetic_vector_list = []
synth_avg_len_list = []

previous_word = None
for answer in json_data:
    if answer['creator'] == 'human':
        answer_text = answer['answer']
        words = answer_text.split()
        vector_norms = calculate_vector_norms(words, model)
        if vector_norms:
            average_norm = np.mean(vector_norms)
            human_avg_len_list.append(average_norm)
        for word in words:
            word_embedding = get_word_embedding(word, model)
            if word_embedding is not None:
                current_word = word
                if previous_word != current_word:
                    #Do something with the word embedding or token
                    vector_distance = calculate_distance(current_word, previous_word)
                    human_vector_list.append(vector_distance)
            
                    #print(vector_distance)
                    #print(current_word, word_embedding)
                previous_word = current_word
    if answer['creator'] == 'ai':
        answer_text = answer['answer']
        words = answer_text.split()
        vector_norms = calculate_vector_norms(words, model)
        if vector_norms:
            average_norm = np.mean(vector_norms)
            synth_avg_len_list.append(average_norm)
        for word in words:
            word_embedding = get_word_embedding(word, model)
            if word_embedding is not None:
                current_word = word
                if previous_word != current_word:
                    #Do something with the word embedding or token
                    vector_distance = calculate_distance(current_word, previous_word)
                    synthetic_vector_list.append(vector_distance)
                    #print(vector_distance)
                    #print(current_word, word_embedding)
                previous_word = current_word




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
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(text1_distances, label='Text 1 Distances', color='b')
plt.plot(text2_distances, label='Text 2 Distances', color='r')
plt.title('Average Euclidean Distances for Word Pairs')
plt.xlabel('Number of Answers')
plt.ylabel('Average Distance')
plt.legend()
plt.grid(True)

# Plotting the derivatives of the distances
plt.subplot(1, 3, 2)
plt.plot(text1_derivatives, label='Derivative of Text 1 Distances', color='b')
plt.plot(text2_derivatives, label='Derivative of Text 2 Distances', color='r')
plt.title('Derivatives of Distances')
plt.xlabel('Number of Tokens')
plt.ylabel('Rate of Change in Distance')
plt.legend()
plt.grid(True)
# print(text2_derivatives)
# print("READTHIIISSSS")
# print(human_vector_list)
# print(synthetic_vector_list)

# Plotting the vector distances
plt.subplot(1, 3, 3)
plt.plot(human_avg_len_list, label='Human', color='b')
plt.plot(synth_avg_len_list, label='Synthetic', color='r')
plt.title('Word Embedding Vector Distances')
plt.xlabel('Word Index')
plt.ylabel('Vector Distance')
plt.legend()
plt.grid(True)


plt.tight_layout()

plt.show()
