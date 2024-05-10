import json
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.spatial import distance  # Import the module
from scipy.spatial.distance import cosine  # Import the cosine function directly


# Load the Word2Vec model
#Todo! Add URL to source
#model_path = 'models/GoogleNews-vectors-negative300.bin' 
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
llama3_prompt1_json_data = read_json_file()
chatGpt3_prompt1_json_data = read_json_file('Test-Data/output_chatGpt_prompt1_Student.json')

prompt_2_human_answers_json_data = read_json_file('Test-Data/prompt_2_human_output.json')
llama3_prompt2_json_data = read_json_file('Test-Data/prompt_2_ai_output.json')
chatGpt3_prompt2_json_data = read_json_file('Test-Data/output_chatGpt_prompt2_Student.json')

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
    norms = [np.linalg.norm(model[word]) for word in words if word in model]
    return norms

def smoothing_average(values, window_size):
    result = []
    moving_sum = sum(values[:window_size])
    result.append(moving_sum/window_size)
    for i in range(len(values)- window_size):
        moving_sum+= (values[i + window_size] - values[i])
        result.append(moving_sum/window_size)
        
    weights = np.repeat(1.0, window_size) / window_size
    #sma = np.convolve(values , weights, 'valid')
    return result

def calculate_distance_and_more(creator, json_data):
    avg_len_list = []
    vector_list = []
    cosine_list = []
    variance_list = []
    cosine_variance_list = []
    distances = []
    previous_word = None
    for answer in json_data:
        if answer['creator'] == creator:
            answer_text = answer['answer']
            dist = calculate_distances_for_text(answer_text, model)
            if dist is not None:
                distances.append(dist)
            words = answer_text.split()
            vector_norms = calculate_vector_norms(words, model)
            if vector_norms:
                average_norm = np.mean(vector_norms)
                avg_len_list.append(average_norm)
            for word in words:
                word_embedding = get_word_embedding(word, model)
                if word_embedding is not None:
                    current_word = word
                    if previous_word != current_word:
                        #Do something with the word embedding or token
                        vector_distance = calculate_distance(current_word, previous_word)
                        vector_list.append(vector_distance)
                        if previous_word in model and current_word in model:
                            sim = cosine_similarity(model[previous_word], model[current_word])
                            cosine_list.append(sim) 
                            variance = np.var(cosine_list)   
                    previous_word = current_word
            cosine_variance_list.append(variance)
    return avg_len_list, vector_list, cosine_list, cosine_variance_list, distances

# * Calculate the distances and more
promp1_human_avg_len_list, prompt1_human_vector_list, prompt1_human_cosine_list, prompt1_human_cosine_variances, prompt1_human_distances = calculate_distance_and_more('human', llama3_prompt1_json_data)
prompt1_llama3_avg_len_list, prompt1_llama3_vector_list, prompt1_llama3_cosine_list, prompt1_llama3_cosine_variances, prompt1_llama3_distances = calculate_distance_and_more('ai', llama3_prompt1_json_data)
prompt1_gpt3_avg_len_list, prompt1_gpt3_vector_list, prompt1_gpt3_cosine_list, prompt1_gpt3_cosine_variances, prompt1_gpt3_distances = calculate_distance_and_more('ai', chatGpt3_prompt1_json_data)

prompt2_human_avg_len_list, prompt2_human_vector_list, prompt2_human_cosine_list, prompt2_human_cosine_variances, prompt2_human_distances = calculate_distance_and_more('human', prompt_2_human_answers_json_data)
prompt2_llama3_avg_len_list, prompt2_llama3_vector_list, prompt2_llama3_cosine_list, prompt2_llama3_cosine_variances, prompt2_llama3_distances = calculate_distance_and_more('ai', llama3_prompt2_json_data)
prompt2_gpt3_avg_len_list, prompt2_gpt3_vector_list, prompt2_gpt3_cosine_list, prompt2_gpt3_cosine_variances, prompt2_gpt3_distances = calculate_distance_and_more('ai', chatGpt3_prompt2_json_data)         

# Calculate derivatives of the distance arrays
prompt1_human_derivatives = np.gradient(prompt1_human_distances)
prompt1_llama3_derivatives = np.gradient(prompt1_llama3_distances)
prompt1_gpt3_derivatives = np.gradient(prompt1_gpt3_distances)

prompt2_human_derivatives = np.gradient(prompt2_human_distances)
prompt2_llama3_derivatives = np.gradient(prompt2_llama3_distances)
prompt2_gpt3_derivatives = np.gradient(prompt2_gpt3_distances)

prompt1_human_smoo_avg_list = smoothing_average(promp1_human_avg_len_list, 10)
prompt1_llama3_smoo_avg_list = smoothing_average(prompt1_llama3_avg_len_list,10)
prompt1_gpt3_smoo_avg_list = smoothing_average(prompt1_gpt3_avg_len_list,10)

prompt2_human_smoo_avg_list = smoothing_average(prompt2_human_avg_len_list, 10)
prompt2_llama3_smoo_avg_list = smoothing_average(prompt2_llama3_avg_len_list,10)
prompt2_gpt3_smoo_avg_list = smoothing_average(prompt2_gpt3_avg_len_list,10)

# * Plotting the original distance data
plt.figure(figsize=(20, 8))
plt.subplot(2, 4, 1)
plt.plot(prompt1_human_distances, label='human Distances', color='b')
plt.plot(prompt1_llama3_distances, label='Llama3 Distances', color='r')
plt.plot(prompt1_gpt3_distances, label='GPT3 Prompt 1 Distances', color='g')
plt.plot(smoothing_average(prompt1_human_distances,5), label='Human - Smoothed Average', linestyle='--', color='black')
plt.plot(smoothing_average(prompt1_llama3_distances, 5), label='Llama3 - Smoothed Average', color='black')
plt.title('Average Euclidean Distances for Word Pairs')
plt.xlabel('Number of Answers')
plt.ylabel('Average Distance')
plt.legend()
plt.grid(True)

# *  Plotting the Cosines of the Embedded Tokens
plt.subplot(2, 4, 2)
#plt.plot(human_derivatives, label='Derivative of Text 1 Distances', color='b')
#plt.plot(synthetic_derivatives, label='Derivative of Text 2 Distances', color='r')
plt.plot(smoothing_average(prompt1_human_cosine_list, 2000), label='Human Cosine Similarities', color='b')
plt.plot(smoothing_average(prompt1_llama3_cosine_list, 2000), label='Llama3 Cosine Similarities', color='r')
plt.plot(smoothing_average(prompt1_gpt3_cosine_list, 2000), label='GPT3 Cosine Similarities', color='g')
plt.title('Smothed Average of Cosine Similarity Scores')
plt.xlabel('Number of Tokens')
plt.ylabel('Cosine Similarity')
plt.legend()
plt.grid(True)

# *  Plotting the Variances
plt.subplot(2, 4, 3)
plt.plot(smoothing_average(prompt1_human_cosine_variances, 1), label='Human Cosine Similarities', color='b')
plt.plot(smoothing_average(prompt1_llama3_cosine_variances, 1), label='Llama3 Cosine Similarities', color='r')
plt.plot(smoothing_average(prompt1_gpt3_cosine_variances, 1), label='GPT3 Cosine Similarities', color='g')
plt.title('Variance')
plt.xlabel('Number of Answers')
plt.ylabel('Avg. Squared difference between record and mean')
plt.legend()
plt.grid(True)

# * Plotting the vector distances
plt.subplot(2, 4, 4)
plt.plot(promp1_human_avg_len_list, label='Human', color='b')
plt.plot(prompt1_llama3_avg_len_list, label='Llama3', color='r')
plt.plot(prompt1_gpt3_avg_len_list, label='GPT3', color='g')
plt.plot(prompt1_human_smoo_avg_list, label='Human - Smoothed Average', linestyle='--', color='black')
plt.plot(prompt1_llama3_smoo_avg_list, label='Llama3 - Smoothed Average', color='black')
plt.plot(prompt1_gpt3_smoo_avg_list, label='GPT3 - Smoothed Average', color='black')
plt.title('Word Embedding Vector Distances')
plt.xlabel('Word Index')
plt.ylabel('Vector Distance')
plt.legend()
plt.grid(True)

# Prompt 2
plt.subplot(2, 4, 5)
plt.plot(prompt2_llama3_distances, label='Llama3 Distances', color='r')
plt.plot(prompt2_gpt3_distances, label='GPT3 Prompt 2 Distances', color='g')
plt.plot(prompt2_human_distances, label='human Distances', color='b')
plt.title('Average Euclidean Distances for Word Pairs')
plt.xlabel('Number of Answers')
plt.ylabel('Average Distance')
plt.legend()
plt.grid(True)

plt.subplot(2, 4, 6)
plt.plot(smoothing_average(prompt2_human_cosine_list, 2000), label='Human Cosine Similarities', color='b')
plt.plot(smoothing_average(prompt2_llama3_cosine_list, 2000), label='Llama3 Cosine Similarities', color='r')
plt.plot(smoothing_average(prompt2_gpt3_cosine_list, 2000), label='GPT3 Cosine Similarities', color='g')
plt.title('Smothed Average of Cosine Similarity Scores')
plt.xlabel('Number of Tokens')
plt.ylabel('Cosine Similarity')
plt.legend()
plt.grid(True)

# *  Plotting the Cosines of the Embedded Tokens
plt.subplot(2, 4, 7)
plt.plot(smoothing_average(prompt2_human_cosine_list, 2000), label='Human Cosine Similarities', color='b')
plt.plot(smoothing_average(prompt2_llama3_cosine_list, 2000), label='Llama3 Cosine Similarities', color='r')
plt.plot(smoothing_average(prompt2_gpt3_cosine_list, 2000), label='GPT3 Cosine Similarities', color='g')
plt.title('Smothed Average of Cosine Similarity Scores')
plt.xlabel('Number of Tokens')
plt.ylabel('Cosine Similarity')
plt.legend()
plt.grid(True)

# *  Plotting the Variances
plt.subplot(2, 4, 8)
plt.plot(smoothing_average(prompt2_human_cosine_variances, 1), label='Human Cosine Similarities', color='b')
plt.plot(smoothing_average(prompt2_llama3_cosine_variances, 1), label='Llama3 Cosine Similarities', color='r')
plt.plot(smoothing_average(prompt2_gpt3_cosine_variances, 1), label='GPT3 Cosine Similarities', color='g')
plt.title('Variance')
plt.xlabel('Number of Answers')
plt.ylabel('Avg. Squared difference between record and mean')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

