import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.spatial import distance  # Import the module
from scipy.spatial.distance import cosine  # Import the cosine function directly

# If set to True, the data will be recalculated and saved to a file
# If set to False, the data will be loaded from the file
GENERATE_NEW_DATA = False
MODEL = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
# MODEL = KeyedVectors.load_word2vec_format("models/crawl-300d-2M.vec")
# MODEL = KeyedVectors.load_word2vec_format("models/wiki-news-300d-1M.vec")


class CalculationsObject:
    """
    Class to store the calculated data for each model
    Contains all the data calculated for a model
    """

    def __init__(self, avg_len_list,
                 vector_list,
                 cosine_list,
                 cosine_variance_list,
                 cosine_variance_per_answer,
                 distances,
                 covariances,
                 word_embeddings):

        self.avg_len_list = avg_len_list
        self.vector_list = vector_list
        self.cosine_list = cosine_list
        self.cosine_variance_list = cosine_variance_list
        self.cosine_variance_per_answer = cosine_variance_per_answer
        self.distances = distances
        self.covariances = covariances
        self.word_embeddings = word_embeddings
        self.derivatives = np.gradient(distances) if len(distances) > 1 else None
        self.smooth_avg_list = smoothing_average(avg_len_list, 10)


# def create_large_distance_plot():
#     plt.figure(figsize=(20, 8))
#     plt.subplot(1, 1, 1)
#     plt.plot(smoothing_average(prompt1_human.distances, 5), label='Human', color='b')
#     # plt.plot(smoothing_average(prompt1_llama2.distances, 5), label='Llama2 Student', color='black')
#     plt.plot(smoothing_average(prompt1_llama3_student.distances, 5), label='Llama3 Student', color='r')
#     plt.plot(smoothing_average(prompt1_gpt3_student.distances, 5), label='GPT3 Student', color='g')
#     plt.plot(smoothing_average(prompt1_gpt3_plain.distances, 5), label='GPT3 Plain', color='y')
#     plt.plot(smoothing_average(prompt1_gpt3_humanlike.distances, 5), label='GPT3 Humanlike', color='purple')
#     plt.plot(smoothing_average(prompt1_gpt4_student.distances, 5), label='GPT4 Student', color='orange')
#     plt.title('Prompt 1 - Avg Euclidean Distances')
#     plt.xlabel('Number of Answers')
#     plt.ylabel('Average Distance')
#     plt.ylim(2.6, 3.2)
#     plt.xlim(0, 100)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


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


def calculate_distances_for_text(text, input_model):
    word_pairs = generate_word_pairs(text)
    distances = [distance.euclidean(input_model[word1], input_model[word2])
                 for word1, word2 in word_pairs if word1 in input_model and word2 in input_model]
    if distances:
        return np.mean(distances)
    return None


def cosine_similarity(vector1, vector2):
    cos_dist = cosine(vector1, vector2)
    cos_similarity = 1 - cos_dist
    return cos_similarity


def calculate_pairwise_cosine_similarities(words, input_model):
    similarities = []
    vectors = [input_model[word] for word in words if word in input_model]

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = cosine_similarity(vectors[i], vectors[j])
            similarities.append(sim)

    return similarities


def calculate_variance_of_cosine_similarities(text, input_model):
    words = text.split()
    similarities = calculate_pairwise_cosine_similarities(words, input_model)

    if similarities:  # Check if there are similarities computed
        variance = np.var(similarities)
        return variance
    else:
        return None


def get_word_embedding(token, input_model):
    try:
        embedding = input_model[token]
        return embedding
    except KeyError:
        # If the token is not found in the model's vocabulary, return None or a random vector
        return None


def calculate_distance(current_word, previous_word):
    if previous_word in MODEL and current_word in MODEL:
        return distance.euclidean(MODEL[previous_word], MODEL[current_word])
    else:
        return None


def calculate_vector_norms(words, input_model):
    """
    Calculates the L2 norm of the word vectors in the model for the words in the words list.
    """
    norms = [np.linalg.norm(input_model[word]) for word in words if word in input_model]
    return norms


def smoothing_average(values, window_size):
    result = []
    moving_sum = sum(values[:window_size])
    result.append(moving_sum / window_size)
    for i in range(len(values) - window_size):
        moving_sum += (values[i + window_size] - values[i])
        result.append(moving_sum / window_size)

    # weights = np.repeat(1.0, window_size) / window_size
    # sma = np.convolve(values , weights, 'valid')
    return result


def calculate_distance_and_more(model_name, json_data, is_eli5=False):
    avg_len_list = []
    vector_list = []
    cosine_list = []
    cosine_variance_list = []
    distances = []
    covariances_per_answer = []
    cosine_variance_per_answer = []
    word_embeddings = []

    for answer in json_data:
        if is_eli5:
            answer_text = answer[model_name]
        else:
            if answer['creator'] != model_name:
                continue
            answer_text = answer['answer']

        words = answer_text.split()

        # Calculate distances
        dist = calculate_distances_for_text(answer_text, MODEL)
        if dist is not None:
            distances.append(dist)

        vector_norms = []
        i = 0
        skip_increment = False
        covariances = []
        while i < len(words) - 1:
            current_word = words[i]
            if current_word in MODEL:
                word_embeddings.append(MODEL[current_word])
                for j in range(i + 1, len(words)):
                    next_word = words[j]
                    if next_word in MODEL:
                        vector_norms.append(np.linalg.norm(MODEL[current_word]))

                        # Calculate vector distance and cosine similarity
                        vector_list.append(distance.euclidean(MODEL[current_word], MODEL[next_word]))
                        cosine_list.append(cosine_similarity(MODEL[current_word], MODEL[next_word]))

                        # Calculate covariances
                        current_word_embedding, next_word_embedding = MODEL[current_word], MODEL[next_word]
                        mean_current_word, mean_next_word = np.mean(current_word_embedding), np.mean(
                            next_word_embedding)
                        deviation_current_word, deviation_next_word = current_word_embedding - mean_current_word, next_word_embedding - mean_next_word
                        euclid_covariance = np.mean(deviation_current_word * deviation_next_word)
                        covariances.append(euclid_covariance)

                        if cosine_list:
                            variance = np.var(cosine_list)
                            cosine_variance_list.append(variance)

                        if vector_norms:
                            avg_len_list.append(np.mean(vector_norms))

                        i = j
                        skip_increment = True
                        break
            if not skip_increment:
                i += 1
            else:
                skip_increment = False

        covariances_per_answer.append(np.mean(covariances))
        cosine_variance_per_answer.append(np.mean(cosine_variance_list))

    return avg_len_list, vector_list, cosine_list, cosine_variance_list, distances, covariances_per_answer, cosine_variance_per_answer, word_embeddings


def get_prompt1_calculations():
    """
    Creates the objects containing the calculated data for prompt 1
    It assigns the objects to global variables accessible from the entire script
    """

    data_file = "pickles/prompt1_data.pkl"
    # Check if the data file exists
    if os.path.exists(data_file) and not GENERATE_NEW_DATA:
        try:
            # load the data from the file
            with open(data_file, "rb") as file:
                data = pickle.load(file)
            return data
        except (pickle.UnpicklingError, EOFError, ImportError, IndexError) as e:
            print(f"Error loading the data file: {e}")

    # If the data file does not exist, generate the data
    llama3_prompt1_json_data = read_json_file('Test-Data/combined.json')
    prompt1_llama2_student_json_data = read_json_file('Test-Data/prompt1_llama2_student.json')
    prompt1_chatGpt3_student_json_data = read_json_file('Test-Data/output_chatGpt_prompt1_Student.json')
    prompt1_chatGpt3_plain_json_data = read_json_file('Test-Data/prompt1_gpt3_plain.json')
    prompt1_chatGpt3_humanlike_json_data = read_json_file('Test-Data/prompt1_gpt3_humanlike.json')
    prompt1_chatGpt4_student_json_data = read_json_file('Test-Data/prompt1_gpt4_student.json')

    prompt1_human = CalculationsObject(*calculate_distance_and_more('human', llama3_prompt1_json_data))
    prompt1_llama2_student = CalculationsObject(*calculate_distance_and_more('ai', prompt1_llama2_student_json_data))
    prompt1_llama3_student = CalculationsObject(*calculate_distance_and_more('ai', llama3_prompt1_json_data))
    prompt1_gpt3_student = CalculationsObject(*calculate_distance_and_more('ai', prompt1_chatGpt3_student_json_data))
    prompt1_gpt3_plain = CalculationsObject(*calculate_distance_and_more('ai', prompt1_chatGpt3_plain_json_data))
    prompt1_gpt3_humanlike = CalculationsObject(
        *calculate_distance_and_more('ai', prompt1_chatGpt3_humanlike_json_data))
    prompt1_gpt4_student = CalculationsObject(*calculate_distance_and_more('ai', prompt1_chatGpt4_student_json_data))

    # Save the data to a file
    with open(data_file, "wb") as file:
        data = (prompt1_human, prompt1_llama2_student, prompt1_llama3_student, prompt1_gpt3_student, prompt1_gpt3_plain,
                prompt1_gpt3_humanlike, prompt1_gpt4_student)
        pickle.dump(data, file)

    return data


def get_prompt2_calculations():
    """
    Creates the objects containing the calculated data for prompt 2
    It assigns the objects to global variables accessible from the entire script
    """

    data_file = "pickles/prompt2_data.pkl"

    # Check if the data file exists
    if os.path.exists(data_file) and not GENERATE_NEW_DATA:
        try:
            # load the data from the file
            with open(data_file, "rb") as file:
                data = pickle.load(file)
            return data
        except (pickle.UnpicklingError, EOFError, ImportError, IndexError) as e:
            print(f"Error loading the data file: {e}")

    prompt_2_human_answers_json_data = read_json_file('Test-Data/prompt_2_human_output.json')
    prompt2_llama3_json_data = read_json_file('Test-Data/prompt_2_ai_output.json')
    prompt2_llama2_student_json_data = read_json_file('Test-Data/prompt2_llama2_student.json')
    prompt2_chatGpt3_json_data = read_json_file('Test-Data/output_chatGpt_prompt2_Student.json')
    prompt2_chatGpt3_plain_json_data = read_json_file('Test-Data/prompt2_gpt3_plain.json')
    prompt2_chatGpt3_humanlike_json_data = read_json_file('Test-Data/prompt2_gpt3_humanlike.json')
    prompt2_chatGpt4_student_json_data = read_json_file('Test-Data/prompt2_gpt4_student.json')

    prompt2_human = CalculationsObject(*calculate_distance_and_more('human', prompt_2_human_answers_json_data))
    prompt2_llama2_student = CalculationsObject(*calculate_distance_and_more('ai', prompt2_llama2_student_json_data))
    prompt2_llama3_student = CalculationsObject(*calculate_distance_and_more('ai', prompt2_llama3_json_data))
    prompt2_gpt3_student = CalculationsObject(*calculate_distance_and_more('ai', prompt2_chatGpt3_json_data))
    prompt2_gpt3_plain = CalculationsObject(*calculate_distance_and_more('ai', prompt2_chatGpt3_plain_json_data))
    prompt2_gpt3_humanlike = CalculationsObject(
        *calculate_distance_and_more('ai', prompt2_chatGpt3_humanlike_json_data))
    prompt2_gpt4_student = CalculationsObject(*calculate_distance_and_more('ai', prompt2_chatGpt4_student_json_data))

    # Save the data to a file
    with open(data_file, "wb") as file:
        data = (prompt2_human, prompt2_llama2_student, prompt2_llama3_student, prompt2_gpt3_student, prompt2_gpt3_plain,
                prompt2_gpt3_humanlike, prompt2_gpt4_student)
        pickle.dump(data, file)

    return data


def get_eli5_calculations():
    data_file = "pickles/eli5_data.pkl"
    # Check if the data file exists
    if os.path.exists(data_file) and not GENERATE_NEW_DATA:
        try:
            # load the data from the file
            with open(data_file, "rb") as file:
                data = pickle.load(file)
            return data
        except (pickle.UnpicklingError, EOFError, ImportError, IndexError) as e:
            print(f"Error loading the data file: {e}")

    eli5_json_data = read_json_file('Test-Data/eli5_2.json')

    eli5_human = CalculationsObject(*calculate_distance_and_more('human', eli5_json_data, True))
    eli5_llama2 = CalculationsObject(*calculate_distance_and_more('llama2', eli5_json_data, True))
    eli5_llama3 = CalculationsObject(*calculate_distance_and_more('llama3', eli5_json_data, True))
    eli5_chatGpt3 = CalculationsObject(*calculate_distance_and_more('chatGpt3', eli5_json_data, True))

    # Save the data to a file
    with open(data_file, "wb") as file:
        data = (eli5_human, eli5_llama2, eli5_llama3, eli5_chatGpt3)
        pickle.dump(data, file)

    return data


def get_openqa_calculations():
    data_file = "pickles/openqa_data.pkl"
    # Check if the data file exists
    if os.path.exists(data_file) and not GENERATE_NEW_DATA:
        try:
            # load the data from the file
            with open(data_file, "rb") as file:
                data = pickle.load(file)
            return data
        except (pickle.UnpicklingError, EOFError, ImportError, IndexError) as e:
            print(f"Error loading the data file: {e}")

    openqa_json_data = read_json_file('Test-Data/open_qa.json')

    openqa_human = CalculationsObject(*calculate_distance_and_more('human', openqa_json_data, True))
    # openqa_llama2 = CalculationsObject(*calculate_openqa_distances_and_more('llama2', openqa_json_data))
    # openqa_llama3 = CalculationsObject(*calculate_openqa_distances_and_more('llama3', openqa_json_data))
    openqa_chatGpt3 = CalculationsObject(*calculate_distance_and_more('chatGpt3', openqa_json_data, True))

    # Save the data to a file
    with open(data_file, "wb") as file:
        data = (openqa_human, openqa_chatGpt3)
        pickle.dump(data, file)

    return data


def create_prompt1_plots():
    """
    Creates the plots for the data of prompt 1
    """

    prompt1_human, prompt1_llama2_student, prompt1_llama3_student, prompt1_gpt3_student, prompt1_gpt3_plain, prompt1_gpt3_humanlike, prompt1_gpt4_student = get_prompt1_calculations()

    # * Plotting the original distance data
    plt.figure(1, figsize=(20, 8))
    plt.suptitle("Prompt 1")
    plt.subplot(1, 3, 1)
    plt.plot(smoothing_average(prompt1_human.distances, 10), label='Human', color='blue')
    plt.plot(smoothing_average(prompt1_llama2_student.distances, 10), label='Llama2 Student', color='black')
    plt.plot(smoothing_average(prompt1_llama3_student.distances, 10), label='Llama3 Student', color='red')
    plt.plot(smoothing_average(prompt1_gpt3_student.distances, 10), label='GPT3 Student', color='green')
    plt.plot(smoothing_average(prompt1_gpt4_student.distances, 10), label='GPT4 Student', color='orange')
    plt.title('Avg. Euclidean Distance')
    plt.xlabel('Answers')
    plt.ylabel('Average Distance')
    # plt.ylim(2.6, 3.1)
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(prompt1_human.covariances, label='Human', color='blue')
    plt.plot(prompt1_llama2_student.covariances, label='Llama2 Student', color='black')
    plt.plot(prompt1_llama3_student.covariances, label='Llama3 Student', color='red')
    plt.plot(prompt1_gpt3_student.covariances, label='GPT3 Student', color='green')
    plt.plot(prompt1_gpt4_student.covariances, label='GPT4 Student', color='orange')
    plt.title('Covariances')
    plt.xlabel('Answers')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(smoothing_average(prompt1_human.cosine_variance_list, 1), label='Human', color='blue')
    plt.plot(smoothing_average(prompt1_llama2_student.cosine_variance_list, 1), label='Llama2 Student', color='black')
    plt.plot(smoothing_average(prompt1_llama3_student.cosine_variance_list, 1), label='Llama3 Student', color='red')
    plt.plot(smoothing_average(prompt1_gpt3_student.cosine_variance_list, 1), label='GPT3 Student', color='green')
    plt.plot(smoothing_average(prompt1_gpt4_student.cosine_variance_list, 1), label='GPT4 Student', color='orange')
    plt.title('Cosine variance')
    plt.xlabel('Tokens')
    plt.ylabel('Variance of Cosine Similarities')
    plt.ylim(0.011, 0.022)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()


def create_prompt2_plots():
    """
    Creates the plots for the data of prompt 2
    """

    prompt2_human, prompt2_llama2_student, prompt2_llama3_student, prompt2_gpt3_student, prompt2_gpt3_plain, prompt2_gpt3_humanlike, prompt2_gpt4_student = get_prompt2_calculations()

    plt.figure(2, figsize=(20, 8))
    plt.suptitle("Prompt 2")
    plt.subplot(1, 3, 1)
    plt.plot(smoothing_average(prompt2_human.distances, 10), label='Human', color='blue')
    plt.plot(smoothing_average(prompt2_llama2_student.distances, 10), label='Llama2 Student', color='black')
    plt.plot(smoothing_average(prompt2_llama3_student.distances, 10), label='Llama3 Student', color='red')
    plt.plot(smoothing_average(prompt2_gpt3_student.distances, 10), label='GPT3 Student', color='green')
    plt.plot(smoothing_average(prompt2_gpt4_student.distances, 10), label='GPT4 Student', color='orange')

    plt.title('Avg. Euclidean Distance')
    plt.xlabel('Answers')
    plt.ylabel('Average Distance')
    # plt.ylim(2.6, 3.2)
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(prompt2_human.covariances, label='Human', color='blue')
    plt.plot(prompt2_llama2_student.covariances, label='Llama2 Student', color='black')
    plt.plot(prompt2_llama3_student.covariances, label='Llama3 Student', color='red')
    plt.plot(prompt2_gpt3_student.covariances, label='GPT3 Student', color='green')
    plt.plot(prompt2_gpt4_student.covariances, label='GPT4 Student', color='orange')
    plt.title('Covariances')
    plt.xlabel('Answers')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(smoothing_average(prompt2_human.cosine_variance_list, 1), label='Human', color='blue')
    plt.plot(smoothing_average(prompt2_llama2_student.cosine_variance_list, 1), label='Llama2 Student', color='black')
    plt.plot(smoothing_average(prompt2_llama3_student.cosine_variance_list, 1), label='Llama3 Student', color='red')
    plt.plot(smoothing_average(prompt2_gpt3_student.cosine_variance_list, 1), label='GPT3 Student', color='green')
    plt.plot(smoothing_average(prompt2_gpt4_student.cosine_variance_list, 1), label='GPT4 Student', color='orange')
    plt.title('Variance of Cosine Similarities')
    plt.xlabel('Answers')
    plt.ylabel('Avg. Squared difference between record and mean')
    plt.ylim(0.01, 0.025)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()


def create_eli5_plots():
    eli5_human, eli5_llama2, eli5_llama3, eli5_chatGpt3 = get_eli5_calculations()

    plt.figure(3, figsize=(20, 8))
    plt.suptitle("ELI5")
    plt.subplot(1, 3, 1)
    plt.plot(smoothing_average(eli5_human.distances, 10), label='Human - Smoothed Average', color='blue')
    plt.plot(smoothing_average(eli5_llama2.distances, 10), label='Llama2 - Smoothed Average', color='black')
    plt.plot(smoothing_average(eli5_llama3.distances, 10), label='Llama3 - Smoothed Average', color='red')
    plt.plot(smoothing_average(eli5_chatGpt3.distances, 10), label='GPT3 - Smoothed Average', color='green')
    plt.title('Avg. Euclidean Distance')
    plt.xlabel('Number of Answers')
    plt.ylabel('Average Distance')
    # plt.ylim(2.7, 3.2)
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(smoothing_average(eli5_human.covariances, 5), label='Human', color='blue')
    plt.plot(smoothing_average(eli5_llama2.covariances, 5), label='Llama2', color='black')
    plt.plot(smoothing_average(eli5_llama3.covariances, 5), label='Llama3', color='red')
    plt.plot(smoothing_average(eli5_chatGpt3.covariances, 5), label='GPT3', color='green')
    plt.title('Covariances')
    plt.xlabel('Number of Answers')
    plt.ylabel('Covariance')
    # plt.ylim(0.002, 0.005)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(smoothing_average(eli5_human.cosine_variance_list, 1), label='Human', color='blue')
    plt.plot(smoothing_average(eli5_llama2.cosine_variance_list, 1), label='Llama2', color='black')
    plt.plot(smoothing_average(eli5_llama3.cosine_variance_list, 1), label='Llama3', color='red')
    plt.plot(smoothing_average(eli5_chatGpt3.cosine_variance_list, 1), label='GPT3', color='green')
    plt.title('Variance of Cosine Similarities')
    plt.xlabel('Number of Answers')
    plt.ylabel('Avg. Squared difference between record and mean')
    plt.ylim(0.015, 0.035)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()


def create_openqa_plots():
    openqa_human, openqa_chatGpt3 = get_openqa_calculations()

    plt.figure(4, figsize=(20, 8))
    plt.suptitle("OpenQA")
    plt.subplot(1, 3, 1)
    plt.plot(smoothing_average(openqa_human.distances, 10), label='Human - Smoothed Average', color='blue')
    plt.plot(smoothing_average(openqa_chatGpt3.distances, 10), label='GPT3 - Smoothed Average', color='green')
    plt.title('Avg. Euclidean Distance')
    plt.xlabel('Number of Answers')
    plt.ylabel('Average Distance')
    # plt.ylim(2.8, 3.4)
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(smoothing_average(openqa_human.covariances, 5), label='Human', color='blue')
    plt.plot(smoothing_average(openqa_chatGpt3.covariances, 5), label='GPT3', color='green')
    plt.title('Covariances')
    plt.xlabel('Number of Answers')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(smoothing_average(openqa_human.cosine_variance_list, 1), label='Human', color='blue')
    plt.plot(smoothing_average(openqa_chatGpt3.cosine_variance_list, 1), label='GPT3', color='green')
    plt.title('Variance of Cosine Similarities')
    plt.xlabel('Number of Answers')
    plt.ylabel('Avg. Squared difference between record and mean')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()


def main():
    # To read, calculate and show prompt 1 data uncomment the following lines
    create_prompt1_plots()

    # To read, calculate and show prompt 2 data uncomment the following lines
    # create_prompt2_plots()

    # create_eli5_plots()

    # create_openqa_plots()

    plt.show()  # Needed in the end to show the plots


if __name__ == '__main__':
    main()
