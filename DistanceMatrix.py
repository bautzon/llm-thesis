import json
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.spatial import distance

# Load the Word2Vec model
#Todo! Add URL to source
model_path = 'models/GoogleNews-vectors-negative300.bin' 
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

def cosine_similarity(vector1, vector2):
    cos_dist = cosine(vector1, vector2)
    cos_similarity = 1 - cos_dist
    return cos_similarity 

def calculate_pairwise_cosine_similarities(words, model):
    similarities = []
    vectors = [model[word] for word in words if word in model]
    
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = cosine_similarity(vectors[i], vectors[j])
            similarities.append(sim)
    
    return similarities

def calculate_variance_of_cosine_similarities(text, model):
    words = text.split()
    similarities = calculate_pairwise_cosine_similarities(words, model)
    
    if similarities:  # Check if there are similarities computed
        variance = np.var(similarities)
        return variance
    else:
        return None



# * Process the data
json_data = read_json_file()

# * Initialize Empty Lists
hum_smoo_avg_list =          []
syn_smoo_avg_list =          []
human_vector_list =          []
human_avg_len_list =         []
synthetic_vector_list =      []
synth_avg_len_list =         []
human_cosine_list =          []
synthetic_cosine_list =      []
human_distances =            []
synthetic_distances =        []
human_dist_variances =       []
synthetic_dist_variances =   []
human_cosine_variances =     []
synthetic_cosine_variances=  []


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
human_variance = []
ai_variance = []

def get_word_embedding(token, model):
    try:
        embedding = model[token]
        return embedding
    except KeyError:
        return None  # Return None for missing tokens

def calculate_distance(current_word, previous_word, model):
    if previous_word in model and current_word in model:
        return np.linalg.norm(model[current_word] - model[previous_word])
    else:
        return None

def calculate_vector_norms(words, model):
    return [np.linalg.norm(model[word]) for word in words if word in model]

for answer in json_data:
    words = answer['answer'].split()
    vector_norms = calculate_vector_norms(words, model)
    if vector_norms:
        average_norm = np.mean(vector_norms)
        if answer['creator'] == 'human':
            human_avg_len_list.append(average_norm)
        else:
            synth_avg_len_list.append(average_norm)
    
    vector_distances = []
    previous_word = None
    for word in words:
        word_embedding = get_word_embedding(word, model)
        if word_embedding is not None and previous_word:
            vector_distance = calculate_distance(word, previous_word, model)
            if vector_distance is not None:
                vector_distances.append(vector_distance)
        previous_word = word

    if vector_distances:  # Only calculate variance if there are distances
        calculated_variance = np.var(vector_distances, dtype=np.float64)
        if answer['creator'] == 'human':
            human_dist_variances.append(calculated_variance)
        else:
            synthetic_dist_variances.append(calculated_variance)


# for comparison in json_data:
#     avg_dist1 = calculate_distances_for_text(comparison['prompt'], model)
#     avg_dist2 = calculate_distances_for_text(comparison['answer'], model)
#     if avg_dist1 is not None:
#         text1_distances.append(avg_dist1)
#     if avg_dist2 is not None:
#         text2_distances.append(avg_dist2)

# # Calculate derivatives of the distance arrays
# text1_derivatives = np.gradient(text1_distances)
# text2_derivatives = np.gradient(text2_distances)



# Plotting the original distance data
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(human_variance, label='Humn euclid var', marker='o', color='b')
plt.plot(ai_variance, label='ai euclid var', marker='x', color='r')
plt.title('Euclidean variance')
plt.xlabel('Number of Answers')
plt.ylabel('variance in distance')
plt.legend()
plt.grid(True)

# Plotting the derivatives of the distances
# plt.subplot(1, 3, 2)
# plt.plot(text1_derivatives, label='Derivative of Text 1 Distances', marker='o', linestyle='--', color='b')
# plt.plot(text2_derivatives, label='Derivative of Text 2 Distances', marker='x', linestyle='--', color='r')
# plt.title('Derivatives of Distances')
# plt.xlabel('Number of Tokens')
# plt.ylabel('Rate of Change in Distance')
# plt.legend()
# plt.grid(True)
# print(text2_derivatives)
# print("READTHIIISSSS")
# print(human_vector_list)
# print(synthetic_vector_list)

# Plotting the vector distances
plt.subplot(1, 3, 3)
plt.plot(human_avg_len_list, label='Human', linestyle='--', color='b')
plt.plot(synth_avg_len_list, label='Synthetic', linestyle='--', color='r')
plt.title('Word Embedding Vector Distances')
plt.xlabel('Word Index')
plt.ylabel('Vector Distance')
plt.legend()
plt.grid(True)

human_indices = list(range(1, len(human_dist_variances) + 1))
ai_indices = list(range(1, len(synthetic_dist_variances) + 1))

def moving_average(data, window_size):
    """Calculate the moving average using a window of the specified size."""
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'lort')

plt.figure(figsize=(10, 5))
plt.plot(moving_average(human_dist_variances,5), label='Human-generated', marker='o')
plt.plot(moving_average(synthetic_dist_variances,5), label='AI-generated', marker='x')
plt.title('Comparison of Distance Variances')
plt.xlabel('Answer Index')
plt.ylabel('Variance of Distances')
plt.legend()
plt.grid(True)
plt.show()



plt.tight_layout()

plt.show()


