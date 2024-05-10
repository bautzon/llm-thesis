import json
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.spatial import distance  # Import the module
from scipy.spatial.distance import cosine  # Import the cosine function directly

def create_large_distance_plot():
    plt.figure(figsize=(24, 10))
    plt.subplot(1, 1, 1)
    plt.plot(smoothing_average(prompt1_human_distances, 5), label='Human', color='b')
    plt.plot(smoothing_average(prompt1_llama2_distances, 5), label='Llama2 Student', color='black')
    plt.plot(smoothing_average(prompt1_llama3_distances, 5), label='Llama3 Student', color='r')
    plt.plot(smoothing_average(prompt1_gpt3_student_distances, 5), label='GPT3 Student', color='g')
    plt.plot(smoothing_average(prompt1_gpt3_plain_distances, 5), label='GPT3 Plain', color='y')
    plt.plot(smoothing_average(prompt1_gpt3_humanlike_distances, 5), label='GPT3 Humanlike', color='purple')
    plt.plot(smoothing_average(prompt1_gpt4_student_distances, 5), label='GPT4 Student', color='orange')
    plt.title('Prompt 1 - Avg Euclidean Distances')
    plt.xlabel('Number of Answers')
    plt.ylabel('Average Distance')
    plt.ylim(2.6, 3.2)
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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

def plot_data(subplot, data, labels, colors, title, x_label, y_label):
    plt.subplot(2, 4, subplot)
    for i in range(len(data)):
        plt.plot(data[i], label=labels[i], color=colors[i])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)

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

def create_plots():
    # * Plotting the original distance data
    plt.figure(figsize=(20, 8))

    plot_data(1, 
          [prompt1_human_distances, prompt1_llama3_distances, prompt1_gpt3_student_distances, prompt1_gpt3_plain_distances, smoothing_average(prompt1_human_distances,5), smoothing_average(prompt1_llama3_distances, 5)], 
          ['human', 'Llama3', 'GPT3', 'GPT3 Plain', 'Human - Smoothed Average', 'Llama3 - Smoothed Average'], 
          ['b', 'r', 'g', 'y', 'black', 'black'], 
          'Prompt 1 - Avg. Euclidean Distances', 
          'Number of Answers', 
          'Average Distance')

    plot_data(2,
            [prompt1_human_derivatives, prompt1_llama3_derivatives, prompt1_gpt3_derivatives, smoothing_average(prompt1_human_cosine_list, 2000), smoothing_average(prompt1_llama3_cosine_list, 2000), smoothing_average(prompt1_gpt3_student_cosine_list, 2000)],
            ['Human', 'Llama3', 'Gpt3', 'Human Cosine Similarities', 'Llama3 Cosine Similarities', 'GPT3 Cosine Similarities'],
            ['b', 'r', 'g', 'b', 'r', 'g'],
            'Prompt 1 - Derivatives',
            'Number of Tokens',
            'Cosine Similarity')

    plot_data(3,
            [smoothing_average(prompt1_human_cosine_variances, 1), smoothing_average(prompt1_llama3_cosine_variances, 1), smoothing_average(prompt1_gpt3_student_cosine_variances, 1)],
            ['Human', 'Llama3', 'GPT3'],
            ['b', 'r', 'g'],
            'Prompt 1 - Variance of Cosine Similarities',
            'Number of Answers',
            'Avg. Squared difference between record and mean')

    plot_data(4,
            [promp1_human_avg_len_list, prompt1_llama3_avg_len_list, prompt1_gpt3_student_avg_len_list, prompt1_human_smoo_avg_list, prompt1_llama3_smoo_avg_list, prompt1_gpt3_smoo_avg_list],
            ['Human', 'Llama3', 'GPT3', 'Human - Smoothed Average', 'Llama3 - Smoothed Average', 'GPT3 - Smoothed Average'],
            ['b', 'r', 'g', 'black', 'black', 'black'],
            'Word Embedding Vector Distances',
            'Word Index',
            'Vector Distance')

    plot_data(5,
            [prompt2_human_distances, prompt2_llama3_distances, prompt2_gpt3_distances, prompt2_gpt3_plain_distances],
            ['Human Distances', 'Llama3 Distances', 'GPT3 Distances', 'GPT3 Plain Distances'],
            ['b', 'r', 'g', 'y'],
            'Prompt 2 - Avg. Euclidean Distances',
            'Number of Answers',
            'Average Distance')

    plot_data(6,
            [smoothing_average(prompt2_human_cosine_list, 2000), smoothing_average(prompt2_llama3_cosine_list, 2000), smoothing_average(prompt2_gpt3_cosine_list, 2000)],
            ['Human Cosine Similarities', 'Llama3 Cosine Similarities', 'GPT3 Cosine Similarities'],
            ['b', 'r', 'g'],
            'Prompt 2 - Smothed Avg. of Cosine Similarity Scores',
            'Number of Tokens',
            'Cosine Similarity')

    plot_data(7,
            [smoothing_average(prompt2_human_cosine_variances, 1), smoothing_average(prompt2_llama3_cosine_variances, 1), smoothing_average(prompt2_gpt3_cosine_variances, 1)],
            ['Human', 'Llama3', 'GPT3'],
            ['b', 'r', 'g'],
            'Prompt 2 - Cosine Variance',
            'Number of Answers',
            'Avg. Squared difference between record and mean')

    plot_data(8,
            [prompt2_human_avg_len_list, prompt2_llama3_avg_len_list, prompt2_gpt3_avg_len_list, prompt2_human_smoo_avg_list, prompt2_llama3_smoo_avg_list, prompt2_gpt3_smoo_avg_list],
            ['Human', 'Llama3', 'GPT3', 'Human - Smoothed Average', 'Llama3 - Smoothed Average', 'GPT3 - Smoothed Average'],
            ['b', 'r', 'g', 'black', 'black', 'black'],
            'Prompt 2 - Word Embedding Vector Distances',
            'Word Index',
            'Vector Distance')
         
    plt.tight_layout()
    plt.show()
    

# Load the Word2Vec model
#Todo! Add URL to source
#model_path = 'models/GoogleNews-vectors-negative300.bin' 
model_path = 'models/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# * Read the data
llama3_prompt1_json_data = read_json_file('Test-Data/combined.json')
prompt1_llama2_student_json_data = read_json_file('Test-Data/prompt1_llama2_student.json')
prompt1_chatGpt3_student_json_data = read_json_file('Test-Data/output_chatGpt_prompt1_Student.json')
prompt1_chatGpt3_plain_json_data = read_json_file('Test-Data/prompt1_gpt3_plain.json')
prompt1_chatGpt3_humanlike_json_data = read_json_file('Test-Data/prompt1_gpt3_humanlike.json')
prompt1_chatGpt4_student_json_data = read_json_file('Test-Data/prompt1_gpt4_student.json')

prompt_2_human_answers_json_data = read_json_file('Test-Data/prompt_2_human_output.json')
prompt2_llama3_json_data = read_json_file('Test-Data/prompt_2_ai_output.json')
prompt2_llama2_student_json_data = read_json_file('Test-Data/prompt2_llama2_student.json')
prompt2_chatGpt3_json_data = read_json_file('Test-Data/output_chatGpt_prompt2_Student.json')
prompt2_chatGpt3_plain_json_data = read_json_file('Test-Data/prompt2_gpt3_plain.json')
prompt2_chatGpt3_humanlike_json_data = read_json_file('Test-Data/prompt2_gpt3_humanlike.json')

# * Calculations for prompt 1
promp1_human_avg_len_list, prompt1_human_vector_list, prompt1_human_cosine_list, prompt1_human_cosine_variances, prompt1_human_distances = calculate_distance_and_more('human', llama3_prompt1_json_data)
prompt1_llama3_avg_len_list, prompt1_llama3_vector_list, prompt1_llama3_cosine_list, prompt1_llama3_cosine_variances, prompt1_llama3_distances = calculate_distance_and_more('ai', llama3_prompt1_json_data)
prompt1_llama2_avg_len_list, prompt1_llama2_vector_list, prompt1_llama2_cosine_list, prompt1_llama2_cosine_variances, prompt1_llama2_distances = calculate_distance_and_more('human', prompt1_llama2_student_json_data)
prompt1_gpt3_student_avg_len_list, prompt1_gpt3_student_vector_list, prompt1_gpt3_student_cosine_list, prompt1_gpt3_student_cosine_variances, prompt1_gpt3_student_distances = calculate_distance_and_more('ai', prompt1_chatGpt3_student_json_data)
prompt1_gpt3_plain_avg_len_list, prompt1_gpt3_plain_vector_list, prompt1_gpt3_plain_cosine_list, prompt1_gpt3_plain_cosine_variances, prompt1_gpt3_plain_distances = calculate_distance_and_more('ai', prompt1_chatGpt3_plain_json_data)
prompt1_gpt3_humanlike_avg_len_list, prompt1_gpt3_humanlike_vector_list, prompt1_gpt3_humanlike_cosine_list, prompt1_gpt3_humanlike_cosine_variances, prompt1_gpt3_humanlike_distances = calculate_distance_and_more('ai', prompt1_chatGpt3_humanlike_json_data)
prompt1_gpt4_student_avg_len_list, prompt1_gpt4_student_vector_list, prompt1_gpt4_student_cosine_list, prompt1_gpt4_student_cosine_variances, prompt1_gpt4_student_distances = calculate_distance_and_more('ai', prompt1_chatGpt4_student_json_data)

# * Calculations for prompt 2
prompt2_human_avg_len_list, prompt2_human_vector_list, prompt2_human_cosine_list, prompt2_human_cosine_variances, prompt2_human_distances = calculate_distance_and_more('human', prompt_2_human_answers_json_data)
prompt2_llama3_avg_len_list, prompt2_llama3_vector_list, prompt2_llama3_cosine_list, prompt2_llama3_cosine_variances, prompt2_llama3_distances = calculate_distance_and_more('ai', prompt2_llama3_json_data)
prompt2_llama2_avg_len_list, prompt2_llama2_vector_list, prompt2_llama2_cosine_list, prompt2_llama2_cosine_variances, prompt2_llama2_distances = calculate_distance_and_more('human', prompt2_llama2_student_json_data)
prompt2_gpt3_avg_len_list, prompt2_gpt3_vector_list, prompt2_gpt3_cosine_list, prompt2_gpt3_cosine_variances, prompt2_gpt3_distances = calculate_distance_and_more('ai', prompt2_chatGpt3_json_data)         
prompt2_gpt3_plain_avg_len_list, prompt2_gpt3_plain_vector_list, prompt2_gpt3_plain_cosine_list, prompt2_gpt3_plain_cosine_variances, prompt2_gpt3_plain_distances = calculate_distance_and_more('ai', prompt2_chatGpt3_plain_json_data)

# Calculate derivatives of the distance arrays for prompt 1
prompt1_human_derivatives = np.gradient(prompt1_human_distances)
prompt1_llama2_derivatives = np.gradient(prompt1_llama2_distances)
prompt1_llama3_derivatives = np.gradient(prompt1_llama3_distances)
prompt1_gpt3_derivatives = np.gradient(prompt1_gpt3_student_distances)
prompt1_gpt3_plain_derivatives = np.gradient(prompt1_gpt3_plain_distances)

# Calculate derivatives of the distance arrays for prompt 2
prompt2_human_derivatives = np.gradient(prompt2_human_distances)
prompt2_llama3_derivatives = np.gradient(prompt2_llama3_distances)
prompt2_gpt3_derivatives = np.gradient(prompt2_gpt3_distances)

# Calculate the smoothed average of the distance arrays for prompt 1
prompt1_human_smoo_avg_list = smoothing_average(promp1_human_avg_len_list, 10)
prompt1_llama3_smoo_avg_list = smoothing_average(prompt1_llama3_avg_len_list,10)
prompt1_gpt3_smoo_avg_list = smoothing_average(prompt1_gpt3_student_avg_len_list,10)

# Calculate the smoothed average of the distance arrays for prompt 2
prompt2_human_smoo_avg_list = smoothing_average(prompt2_human_avg_len_list, 10)
prompt2_llama3_smoo_avg_list = smoothing_average(prompt2_llama3_avg_len_list,10)
prompt2_gpt3_smoo_avg_list = smoothing_average(prompt2_gpt3_avg_len_list,10)

# create_plots()
create_large_distance_plot()

"""plt.subplot(2, 4, 1)
plt.plot(prompt1_human_distances, label='human Distances', color='b')
plt.plot(prompt1_llama3_distances, label='Llama3 Distances', color='r')
plt.plot(prompt1_gpt3_distances, label='GPT3 Prompt 1 Distances', color='g')
plt.plot(smoothing_average(prompt1_human_distances,5), label='Human - Smoothed Average', color='black')
plt.plot(smoothing_average(prompt1_llama3_distances, 5), label='Llama3 - Smoothed Average', color='black')
plt.title('Average Euclidean Distances for Word Pairs')
plt.xlabel('Number of Answers')
plt.ylabel('Average Distance')
plt.legend()
plt.grid(True)"""
   

""" # *  Plotting the Cosines of the Embedded Tokens
plt.subplot(2, 4, 2)
#plt.plot(human_derivatives, label='Derivative of Text 1 Distances', color='b')
#plt.plot(synthetic_derivatives, label='Derivative of Text 2 Distances', color='r')
plt.plot(prompt1_gpt3_derivatives, label='Derivative of Text 2 Distances', color='g')
plt.plot(smoothing_average(prompt1_human_cosine_list, 2000), label='Human Cosine Similarities', color='b')
plt.plot(smoothing_average(prompt1_llama3_cosine_list, 2000), label='Llama3 Cosine Similarities', color='r')
plt.plot(smoothing_average(prompt1_gpt3_cosine_list, 2000), label='GPT3 Cosine Similarities', color='g')
plt.title('Smothed Average of Cosine Similarity Scores')
plt.xlabel('Number of Tokens')
plt.ylabel('Cosine Similarity')
plt.legend()
plt.grid(True) """


""" # *  Plotting the Variances
plt.subplot(2, 4, 3)
plt.plot(smoothing_average(prompt1_human_cosine_variances, 1), label='Human Cosine Similarities', color='b')
plt.plot(smoothing_average(prompt1_llama3_cosine_variances, 1), label='Llama3 Cosine Similarities', color='r')
plt.plot(smoothing_average(prompt1_gpt3_cosine_variances, 1), label='GPT3 Cosine Similarities', color='g')
plt.title('Variance')
plt.xlabel('Number of Answers')
plt.ylabel('Avg. Squared difference between record and mean')
plt.legend()
plt.grid(True) """

""" # * Plotting the vector distances
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
plt.grid(True) """


""" # Prompt 2 - Distances
plt.subplot(2, 4, 5)
plt.plot(prompt2_llama3_distances, label='Llama3 Distances', color='r')
plt.plot(prompt2_gpt3_distances, label='GPT3 Prompt 2 Distances', color='g')
plt.plot(prompt2_human_distances, label='human Distances', color='b')
plt.title('Average Euclidean Distances for Word Pairs')
plt.xlabel('Number of Answers')
plt.ylabel('Average Distance')
plt.legend()
plt.grid(True) """


""" # Plotting the Cosines of the Embedded Tokens
plt.subplot(2, 4, 6)
plt.plot(smoothing_average(prompt2_human_cosine_list, 2000), label='Human Cosine Similarities', color='b')
plt.plot(smoothing_average(prompt2_llama3_cosine_list, 2000), label='Llama3 Cosine Similarities', color='r')
plt.plot(smoothing_average(prompt2_gpt3_cosine_list, 2000), label='GPT3 Cosine Similarities', color='g')
plt.title('Smothed Average of Cosine Similarity Scores')
plt.xlabel('Number of Tokens')
plt.ylabel('Cosine Similarity')
plt.legend()
plt.grid(True) """            

""" # *  Plotting the variances
plt.subplot(2, 4, 7)
plt.plot(smoothing_average(prompt2_human_cosine_variances, 1), label='Human Cosine Similarities', color='b')
plt.plot(smoothing_average(prompt2_llama3_cosine_variances, 1), label='Llama3 Cosine Similarities', color='r')
plt.plot(smoothing_average(prompt2_gpt3_cosine_variances, 1), label='GPT3 Cosine Similarities', color='g')
plt.title('Variance')
plt.xlabel('Number of Answers')
plt.ylabel('Avg. Squared difference between record and mean')
plt.legend()
plt.grid(True) """


""" # * Plotting the vector distances
plt.subplot(2, 4, 8)
plt.plot(prompt2_human_avg_len_list, label='Human', color='b')
plt.plot(prompt2_llama3_avg_len_list, label='Llama3', color='r')
plt.plot(prompt2_gpt3_avg_len_list, label='GPT3', color='g')
plt.plot(prompt2_human_smoo_avg_list, label='Human - Smoothed Average', linestyle='--', color='black')
plt.plot(prompt2_llama3_smoo_avg_list, label='Llama3 - Smoothed Average', color='black')
plt.plot(prompt2_gpt3_smoo_avg_list, label='GPT3 - Smoothed Average', color='black')
plt.title('Word Embedding Vector Distances')
plt.xlabel('Word Index')
plt.ylabel('Vector Distance')
plt.legend()
plt.grid(True) """
