import json
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.spatial import distance  # Import the module
from scipy.spatial.distance import cosine  # Import the cosine function directly

class CalculationsObject:
    """
    Class to store the calculated data for each model
    Contains all the data calculated for a model
    """
    def __init__(self, avg_len_list, vector_list, cosine_list, cosine_variance_list, distances):
        self.avg_len_list = avg_len_list
        self.vector_list = vector_list
        self.cosine_list = cosine_list
        self.cosine_variance_list = cosine_variance_list
        self.distances = distances
        self.derivatives = np.gradient(distances)
        self.smooth_avg_list = smoothing_average(avg_len_list, 10)

def create_large_distance_plot():
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 1, 1)
    plt.plot(smoothing_average(prompt1_human.distances, 5), label='Human', color='b')
    #plt.plot(smoothing_average(prompt1_llama2.distances, 5), label='Llama2 Student', color='black')
    plt.plot(smoothing_average(prompt1_llama3_student.distances, 5), label='Llama3 Student', color='r')
    plt.plot(smoothing_average(prompt1_gpt3_student.distances, 5), label='GPT3 Student', color='g')
    plt.plot(smoothing_average(prompt1_gpt3_plain.distances, 5), label='GPT3 Plain', color='y')
    plt.plot(smoothing_average(prompt1_gpt3_humanlike.distances, 5), label='GPT3 Humanlike', color='purple')
    plt.plot(smoothing_average(prompt1_gpt4_student.distances, 5), label='GPT4 Student', color='orange')
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

def create_prompt1_plots():
    """
    Creates the plots for the data of prompt 1
    """
    # * Plotting the original distance data
    plt.figure(1, figsize=(20, 8))
    plt.subplot(2, 2, 1)
    plt.plot(prompt1_human.distances, label='Human', color='b')
    # plt.plot(prompt1_llama2_student.distances, label='Llama2 Student', color='black')
    plt.plot(prompt1_llama3_student.distances, label='Llama3 Student', color='r')
    plt.plot(prompt1_gpt3_humanlike.distances, label='GPT3 Humanlike', color='y')
    plt.plot(prompt1_gpt3_student.distances, label='GPT3 Student', color='purple')
    plt.plot(prompt1_gpt3_plain.distances, label='GPT3 Plain', color='black')
    plt.plot(prompt1_gpt4_student.distances, label='GPT4 Student', color='orange')
    # plt.plot(smoothing_average(prompt1_human_distances,5), label='Human - Smoothed Average', color='black')
    # plt.plot(smoothing_average(prompt1_llama3_distances, 5), label='Llama3 - Smoothed Average', color='black')
    plt.title('Prompt 1 - Avg. Euclidean Distance')
    plt.xlabel('Number of Answers')
    plt.ylabel('Average Distance')
    plt.ylim(2.6, 3.2)
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(prompt1_human.derivatives, label='Human', color='b')
    #plt.plot(prompt1_llama2_student.derivatives, label='Llama2 Student', color='black')
    plt.plot(prompt1_llama3_student.derivatives, label='Llama3 Student', color='r')
    plt.plot(prompt1_gpt3_humanlike.derivatives, label='GPT3 Humanlike', color='y')
    plt.plot(prompt1_gpt3_student.derivatives, label='GPT3 Student', color='purple')
    plt.plot(prompt1_gpt3_plain.derivatives, label='GPT3 Plain', color='black')
    plt.plot(prompt1_gpt4_student.derivatives, label='GPT4 Student', color='orange')
    #plt.plot(smoothing_average(prompt1_human.cosine_list, 2000), label='Human - Smoothed Average', color='black')
    #plt.plot(smoothing_average(prompt1_llama3_student.cosine_list, 2000), label='Llama3 Student - Smoothed Avg.', color='black')
    #plt.plot(smoothing_average(prompt1_gpt3_humanlike.cosine_list, 2000), label='Gpt3 Humanlike - Smoothed Avg.', color='black')
    #plt.plot(smoothing_average(prompt1_gpt3_plain.cosine_list, 2000), label='Gpt3 Plain - Smoothed Avg.', color='black')
    #plt.plot(smoothing_average(prompt1_gpt4_student.cosine_list, 2000), label='Gpt4 Student - Smoothed Avg', color='black')
    plt.title('Prompt 1 - Derivatives')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(smoothing_average(prompt1_human.cosine_variance_list, 1), label='Human', color='b')
    # plt.plot(smoothing_average(prompt1_llama2_student.cosine_variance_list, 1), label='Llama2 Student', color='black')
    plt.plot(smoothing_average(prompt1_llama3_student.cosine_variance_list, 1), label='Llama3 Student', color='r')
    plt.plot(smoothing_average(prompt1_gpt3_humanlike.cosine_variance_list, 1), label='GPT3 Humanlike', color='y')
    plt.plot(smoothing_average(prompt1_gpt3_student.cosine_variance_list, 1), label='GPT3 Student', color='purple')
    plt.plot(smoothing_average(prompt1_gpt3_plain.cosine_variance_list, 1), label='GPT3 Plain', color='black')
    plt.plot(smoothing_average(prompt1_gpt4_student.cosine_variance_list, 1), label='GPT4 Student', color='orange')
    plt.title('Prompt 1 - Variance of Cosine Similarities')
    plt.xlabel('Number of Answers')
    plt.ylabel('Avg. Squared difference between record and mean')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(prompt1_human.avg_len_list, label='Human', color='b')
    # plt.plot(prompt1_llama2_student.avg_len_list, label='Llama2 Student', color='black')
    plt.plot(prompt1_llama3_student.avg_len_list, label='Llama3 Student', color='r')
    plt.plot(prompt1_gpt3_humanlike.avg_len_list, label='GPT3 Humanlike', color='y')
    plt.plot(prompt1_gpt3_student.avg_len_list, label='GPT3 Student', color='purple')
    plt.plot(prompt1_gpt3_plain.avg_len_list, label='GPT3 Plain', color='black')
    plt.plot(prompt1_gpt4_student.avg_len_list, label='GPT4 Student', color='orange')
    #plt.plot(prompt1_human.smooth_avg_list, label='Human - Smoothed Average', linestyle='--', color='black')
    #plt.plot(prompt1_llama3_student.smooth_avg_list, label='Llama3 Student - Smoothed Average', linestyle='--', color='black')
    #plt.plot(prompt1_gpt3_humanlike.smooth_avg_list, label='GPT3 Humanlike - Smoothed Average', linestyle='--', color='black')
    #plt.plot(prompt1_gpt3_student.smooth_avg_list, label='GPT3 Student - Smoothed Average', linestyle='--', color='black')
    #plt.plot(prompt1_gpt3_plain.smooth_avg_list, label='GPT3 Plain - Smoothed Average', linestyle='--', color='black')
    #plt.plot(prompt1_gpt4_student.smooth_avg_list, label='GPT4 Student - Smoothed Average', linestyle='--', color='black')
    plt.title('Prompt 1 - Word Embedding Vector Distances')
    plt.xlabel('Word Index')
    plt.ylabel('Vector Distance')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
def create_prompt2_plots():
    """
    Creates the plots for the data of prompt 2
    """
    plt.figure(2, figsize=(20, 8))
    plt.subplot(2, 2, 1)
    plt.plot(prompt2_human.distances, label='Human', color='b')
    # plt.plot(prompt2_llama2_student.distances, label='Llama2 Student', color='black')
    plt.plot(prompt2_llama3_student.distances, label='Llama3 Student', color='r')
    plt.plot(prompt2_gpt3_student.distances, label='GPT3 Student', color='g')
    plt.plot(prompt2_gpt3_plain.distances, label='GPT3 Plain', color='y')
    plt.plot(prompt2_gpt4_student.distances, label='GPT4 Student', color='orange')
    # plt.plot(smoothing_average(prompt2_human_distances,5), label='Human - Smoothed Average', color='black')
    # plt.plot(smoothing_average(prompt2_llama3_distances, 5), label='Llama3 - Smoothed Average', color='black')
    plt.title('Prompt 2 - Avg. Euclidean Distance')
    plt.xlabel('Number of Answers')
    plt.ylabel('Average Distance')
    plt.ylim(2.6, 3.2)
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(prompt2_human.derivatives, label='Human', color='b')
    #plt.plot(prompt2_llama2_student.derivatives, label='Llama2 Student', color='black')
    plt.plot(prompt2_llama3_student.derivatives, label='Llama3 Student', color='r')
    plt.plot(prompt2_gpt3_student.derivatives, label='GPT3 Student', color='g')
    plt.plot(prompt2_gpt3_plain.derivatives, label='GPT3 Plain', color='y')
    plt.plot(prompt2_gpt4_student.derivatives, label='GPT4 Student', color='orange')
    #plt.plot(smoothing_average(prompt2_human.cosine_list, 2000), label='Human - Smoothed Average', color='black')
    #plt.plot(smoothing_average(prompt2_llama3_student.cosine_list, 2000), label='Llama3 Student - Smoothed Avg.', color='black')
    #plt.plot(smoothing_average(prompt2_gpt3_student.cosine_list, 2000), label='Gpt3 Student - Smoothed Avg.', color='black')
    #plt.plot(smoothing_average(prompt2_gpt3_plain.cosine_list, 2000), label='Gpt3 Plain - Smoothed Avg.', color='black')
    #plt.plot(smoothing_average(prompt2_gpt4_student.cosine_list, 2000), label='Gpt4 Student - Smoothed Avg', color='black')
    plt.title('Prompt 2 - Derivatives')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(smoothing_average(prompt2_human.cosine_variance_list, 1), label='Human', color='b')
    # plt.plot(smoothing_average(prompt2_llama2_student.cosine_variance_list, 1), label='Llama2 Student', color='black')
    plt.plot(smoothing_average(prompt2_llama3_student.cosine_variance_list, 1), label='Llama3 Student', color='r')
    plt.plot(smoothing_average(prompt2_gpt3_student.cosine_variance_list, 1), label='GPT3 Student', color='g')
    plt.plot(smoothing_average(prompt2_gpt3_plain.cosine_variance_list, 1), label='GPT3 Plain', color='y')
    plt.plot(smoothing_average(prompt2_gpt4_student.cosine_variance_list, 1), label='GPT4 Student', color='orange')
    plt.title('Prompt 2 - Variance of Cosine Similarities')
    plt.xlabel('Number of Answers')
    plt.ylabel('Avg. Squared difference between record and mean')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(prompt2_human.avg_len_list, label='Human', color='b')
    # plt.plot(prompt2_llama2_student.avg_len_list, label='Llama2 Student', color='black')
    plt.plot(prompt2_llama3_student.avg_len_list, label='Llama3 Student', color='r')
    plt.plot(prompt2_gpt3_student.avg_len_list, label='GPT3 Student', color='g')
    plt.plot(prompt2_gpt3_plain.avg_len_list, label='GPT3 Plain', color='y')
    plt.plot(prompt2_gpt4_student.avg_len_list, label='GPT4 Student', color='orange')
    #plt.plot(prompt2_human.smooth_avg_list, label='Human - Smoothed Average', linestyle='--', color='black')
    #plt.plot(prompt2_llama3_student.smooth_avg_list, label='Llama3 Student - Smoothed Average', linestyle='--', color='black')
    #plt.plot(prompt2_gpt3_student.smooth_avg_list, label='GPT3 Student - Smoothed Average', linestyle='--', color='black')
    #plt.plot(prompt2_gpt3_plain.smooth_avg_list, label='GPT3 Plain - Smoothed Average', linestyle='--', color='black')
    #plt.plot(prompt2_gpt4_student.smooth_avg_list, label='GPT4 Student - Smoothed Average', linestyle='--', color='black')
    plt.title('Prompt 2 - Word Embedding Vector Distances')
    plt.xlabel('Word Index')
    plt.ylabel('Vector Distance')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
def read_prompt1_data():
    """
    Reads the JSON files containing the answers for prompt 2
    Assigns the data to global variables accessible from the entire script
    """
    global llama3_prompt1_json_data
    llama3_prompt1_json_data = read_json_file('Test-Data/combined.json')
    
    global prompt1_llama2_student_json_data
    prompt1_llama2_student_json_data = read_json_file('Test-Data/prompt1_llama2_student.json')
    
    global prompt1_chatGpt3_student_json_data
    prompt1_chatGpt3_student_json_data = read_json_file('Test-Data/output_chatGpt_prompt1_Student.json')
    
    global prompt1_chatGpt3_plain_json_data
    prompt1_chatGpt3_plain_json_data = read_json_file('Test-Data/prompt1_gpt3_plain.json')
    
    global prompt1_chatGpt3_humanlike_json_data
    prompt1_chatGpt3_humanlike_json_data = read_json_file('Test-Data/prompt1_gpt3_humanlike.json')
    
    global prompt1_chatGpt4_student_json_data
    prompt1_chatGpt4_student_json_data = read_json_file('Test-Data/prompt1_gpt4_student.json')

def read_prompt2_data():
    """
    Reads the JSON files containing the answers for prompt 2
    Assigns the data to global variables accessible from the entire script
    """
    global prompt_2_human_answers_json_data
    prompt_2_human_answers_json_data = read_json_file('Test-Data/prompt_2_human_output.json')

    global prompt2_llama3_json_data
    prompt2_llama3_json_data = read_json_file('Test-Data/prompt_2_ai_output.json')

    global prompt2_llama2_student_json_data
    prompt2_llama2_student_json_data = read_json_file('Test-Data/prompt2_llama2_student.json')

    global prompt2_chatGpt3_json_data
    prompt2_chatGpt3_json_data = read_json_file('Test-Data/output_chatGpt_prompt2_Student.json')

    global prompt2_chatGpt3_plain_json_data
    prompt2_chatGpt3_plain_json_data = read_json_file('Test-Data/prompt2_gpt3_plain.json')

    global prompt2_chatGpt3_humanlike_json_data
    prompt2_chatGpt3_humanlike_json_data = read_json_file('Test-Data/prompt2_gpt3_humanlike.json')

    global prompt2_chatGpt4_student_json_data
    prompt2_chatGpt4_student_json_data = read_json_file('Test-Data/prompt2_gpt4_student.json')

def calculate_prompt1():
    """
    Creates the objects containing the calculated data for prompt 1
    It assigns the objects to global variables accessible from the entire script
    """
    global prompt1_human
    prompt1_human = CalculationsObject(*calculate_distance_and_more('human', llama3_prompt1_json_data))
    
    global prompt1_llama3_student
    prompt1_llama3_student = CalculationsObject(*calculate_distance_and_more('ai', llama3_prompt1_json_data))
    
    #global prompt1_llama2_student
    #prompt1_llama2_student = CalculationsObject(*calculate_distance_and_more('human', prompt1_llama2_student_json_data))
    
    global prompt1_gpt3_student
    prompt1_gpt3_student = CalculationsObject(*calculate_distance_and_more('ai', prompt1_chatGpt3_student_json_data))
    
    global prompt1_gpt3_plain
    prompt1_gpt3_plain = CalculationsObject(*calculate_distance_and_more('ai', prompt1_chatGpt3_plain_json_data))
    
    global prompt1_gpt3_humanlike
    prompt1_gpt3_humanlike = CalculationsObject(*calculate_distance_and_more('ai', prompt1_chatGpt3_humanlike_json_data))
    
    global prompt1_gpt4_student
    prompt1_gpt4_student = CalculationsObject(*calculate_distance_and_more('ai', prompt1_chatGpt4_student_json_data))

def calculate_prompt2():
    """
    Creates the objects containing the calculated data for prompt 2
    It assigns the objects to global variables accessible from the entire script
    """
    global prompt2_human
    prompt2_human = CalculationsObject(*calculate_distance_and_more('human', prompt_2_human_answers_json_data))
    
    # globalprompt2_llama2_student = CalculationsObject(*calculate_distance_and_more('human', prompt2_llama2_student_json_data))
    
    global prompt2_llama3_student
    prompt2_llama3_student = CalculationsObject(*calculate_distance_and_more('ai', prompt2_llama3_json_data))
    
    global prompt2_gpt3_student
    prompt2_gpt3_student = CalculationsObject(*calculate_distance_and_more('ai', prompt2_chatGpt3_json_data))
    
    global prompt2_gpt3_plain
    prompt2_gpt3_plain = CalculationsObject(*calculate_distance_and_more('ai', prompt2_chatGpt3_plain_json_data))
    
    global prompt2_gpt4_student
    prompt2_gpt4_student = CalculationsObject(*calculate_distance_and_more('ai', prompt2_chatGpt4_student_json_data))

def calculate_eli5():
    eli5_json_data = read_json_file('Test-Data/eli5_2.json')
    global eli5_human
    global eli5_llama2
    global eli5_llama3
    global eli5_chatGpt3
    
    eli5_human = CalculationsObject(*calculate_eli5_distances_and_more('human', eli5_json_data))
    eli5_llama2 = CalculationsObject(*calculate_eli5_distances_and_more('llama2', eli5_json_data))
    eli5_llama3 = CalculationsObject(*calculate_eli5_distances_and_more('llama3', eli5_json_data))
    eli5_chatGpt3 = CalculationsObject(*calculate_eli5_distances_and_more('chatGpt3', eli5_json_data))
        
def calculate_eli5_distances_and_more(eli5_model, json_data):
    avg_len_list = []
    vector_list = []
    cosine_list = []
    variance_list = []
    cosine_variance_list = []
    distances = []
    previous_word = None
    for answer in json_data:
        answer_text = answer[eli5_model]
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

def create_eli5_plots():
    plt.figure(1, figsize=(20, 8))
    plt.suptitle("ELI5")
    plt.subplot(2, 2, 1)
    """ plt.plot(eli5_human.distances, label='Human', color='black')
    plt.plot(eli5_llama2.distances, label='Llama2', color='y')
    plt.plot(eli5_llama3.distances, label='Llama3', color='purple')
    plt.plot(eli5_chatGpt3.distances, label='GPT3', color='b') """
    plt.plot(smoothing_average(eli5_human.distances, 10), label='Human - Smoothed Average', color='black')
    plt.plot(smoothing_average(eli5_llama2.distances, 10), label='Llama2 - Smoothed Average', color='y')
    plt.plot(smoothing_average(eli5_llama3.distances, 10), label='Llama3 - Smoothed Average', color='purple')
    plt.plot(smoothing_average(eli5_chatGpt3.distances, 10), label='GPT3 - Smoothed Average', color='b')
    plt.title('Avg. Euclidean Distance')
    plt.xlabel('Number of Answers')
    plt.ylabel('Average Distance')
    plt.ylim(2.5, 3.3)
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(True)
    
    
    plt.subplot(2, 2, 2)
    plt.plot(eli5_human.derivatives, label='Human', color='black')
    plt.plot(eli5_llama2.derivatives, label='Llama2', color='y')
    plt.plot(eli5_llama3.derivatives, label='Llama3', color='purple')
    plt.plot(eli5_chatGpt3.derivatives, label='GPT3', color='b')
    plt.title('Derivatives')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(smoothing_average(eli5_human.cosine_variance_list, 1), label='Human', color='black')
    plt.plot(smoothing_average(eli5_llama2.cosine_variance_list, 1), label='Llama2', color='y')
    plt.plot(smoothing_average(eli5_llama3.cosine_variance_list, 1), label='Llama3', color='purple')
    plt.plot(smoothing_average(eli5_chatGpt3.cosine_variance_list, 1), label='GPT3', color='b')
    plt.title('Variance of Cosine Similarities')
    plt.xlabel('Number of Answers')
    plt.ylabel('Avg. Squared difference between record and mean')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(eli5_human.avg_len_list, label='Human', color='black')
    plt.plot(eli5_llama2.avg_len_list, label='Llama2', color='y')
    plt.plot(eli5_llama3.avg_len_list, label='Llama3', color='purple')
    plt.plot(eli5_chatGpt3.avg_len_list, label='GPT3', color='b')
    plt.title('Word Embedding Vector Distances')
    plt.xlabel('Word Index')
    plt.ylabel('Vector Distance')
    plt.legend()
    plt.grid(True)    

def calculate_openqa():
    global openqa_human
    global openqa_llama2
    global openqa_llama3
    global openqa_chatGpt3
    openqa_json_data = read_json_file('Test-Data/open_qa_cleaned.json')
    
    openqa_human = CalculationsObject(*calculate_openqa_distances_and_more('human', openqa_json_data))
    #openqa_llama2 = CalculationsObject(*calculate_openqa_distances_and_more('llama2', openqa_json_data))
    #openqa_llama3 = CalculationsObject(*calculate_openqa_distances_and_more('llama3', openqa_json_data))
    openqa_chatGpt3 = CalculationsObject(*calculate_openqa_distances_and_more('chatGpt3', openqa_json_data))
    
def create_openqa_plots():
    plt.figure(1, figsize=(20, 8))
    plt.suptitle("OpenQA")
    plt.subplot(2, 2, 1)
    plt.plot(smoothing_average(openqa_human.distances, 10), label='Human - Smoothed Average', color='black')
    plt.plot(smoothing_average(openqa_chatGpt3.distances, 10), label='GPT3 - Smoothed Average', color='b')
    plt.title('Avg. Euclidean Distance')
    plt.xlabel('Number of Answers')
    plt.ylabel('Average Distance')
    plt.ylim(2.5, 3.3)
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(openqa_human.derivatives, label='Human', color='black')
    plt.plot(openqa_chatGpt3.derivatives, label='GPT3', color='b')
    plt.title('Derivatives')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(smoothing_average(openqa_human.cosine_variance_list, 1), label='Human', color='black')
    plt.plot(smoothing_average(openqa_chatGpt3.cosine_variance_list, 1), label='GPT3', color='b')
    plt.title('Variance of Cosine Similarities')
    plt.xlabel('Number of Answers')
    plt.ylabel('Avg. Squared difference between record and mean')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(openqa_human.avg_len_list, label='Human', color='black')
    plt.plot(openqa_chatGpt3.avg_len_list, label='GPT3', color='b')
    plt.title('Word Embedding Vector Distances')
    plt.xlabel('Word Index')
    plt.ylabel('Vector Distance')
    plt.legend()
    plt.grid(True)
    
    
    
# Load the Word2Vec model
model_path = 'models/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# To read, calculate and show prompt 1 data uncomment the following lines
read_prompt1_data()
calculate_prompt1()
create_prompt1_plots()

# To read, calculate and show prompt 2 data uncomment the following lines
#read_prompt2_data()
#calculate_prompt2()
#create_prompt2_plots()

#calculate_eli5()
#create_eli5_plots()

calculate_openqa()
create_openqa_plots()

plt.show() # Needed in the end to show the plots