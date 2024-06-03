import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def plot_perplexities(human_perplexities, llama2_perplexities, llama3_perplexities, gpt3_perplexities, gpt4_perplexities):
    plt.figure(figsize=(12, 6))

    # Plot perplexities
    plt.hist(human_perplexities, bins=20, alpha=0.7, histtype='step', align='right', label='Human Perplexities', color='blue', edgecolor='blue')
    #plt.hist(llama2_perplexities, bins=20, alpha=0.7, histtype='stepfilled', align='left', label='llama2 Perplexities', color='green', edgecolor='green')
    plt.hist(llama3_perplexities, bins=20, alpha=0.7, histtype='stepfilled', align='left', label='llama3 Perplexities', color='orange', edgecolor='orange')
    plt.hist(gpt3_perplexities, bins=20, alpha=0.7, histtype='step', align='mid', label='Gpt3 Perplexities',color='magenta', edgecolor='magenta')
    plt.hist(gpt4_perplexities, bins=20, alpha=0.7, histtype='stepfilled', align='right', label='Gpt4 Perplexities', color='red', edgecolor='red')

    # Add titles and labels
    plt.title('Histogram of Perplexities')
    plt.xlabel('Perplexity')
    plt.ylabel('Frequency')

    # Add a legend
    plt.legend(loc='upper right')

    # Show grid
    plt.grid(axis='y', alpha=0.75)

    # Display the plot
    plt.show()

def plot_KDE(human_perplexities, llama2_perplexities, llama3_perplexities, gpt3_perplexities, gpt4_perplexities):
    plt.figure(figsize=(12, 6))

    # Plot KDEs for perplexities
    sns.kdeplot(human_perplexities, fill=True, label='Human Perplexities', color='blue')
    #sns.kdeplot(llama2_perplexities, fill=True, label='Llama2 Perplexities', color='green')
    sns.kdeplot(llama3_perplexities, fill=True, label='Llama3 Perplexities', color='orange')
    sns.kdeplot(gpt3_perplexities, fill=True, label='Gpt3 Perplexities', color='magenta')
    sns.kdeplot(gpt4_perplexities, fill=True, label='Gpt4 Perplexities', color='red')

    # Add titles and labels
    plt.title('Kernel Density Estimation of Perplexities')
    plt.xlabel('Perplexity')
    plt.ylabel('Density')

    # Add a legend
    plt.legend(loc='upper right')

    # Show grid
    plt.grid(axis='y', alpha=0.75)

    # Display the plot
    plt.show()


def calculate_perplexity(text):
    # Load pre-trained model (weights)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Encode text
    encodings = tokenizer(text, return_tensors='pt')

    # Calculate loss
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)

    return perplexity.item()

<<<<<<< HEAD
# Function to read JSON file
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to adjust the token length of both human and synthetic answers to the minimum length
def adjust_token_lengths(tokenizer, human_answers, llama2_answers, llama3_answers, gpt3_answers, gpt4_answers):
    adjusted_human_answers = []
    adjusted_llama2_answers = []
    adjusted_llama3_answers = []
    adjusted_gpt3_answers = []
    adjusted_gpt4_answers = []

    for human_answer,llama2_answer, llama3_answer, gpt3_answer, gpt4_answer in zip(human_answers, llama2_answers, llama3_answers, gpt3_answers, gpt4_answers):
        human_tokens = tokenizer.tokenize(human_answer)
        llama2_tokens = tokenizer.tokenize(llama3_answer)
        llama3_tokens = tokenizer.tokenize(llama3_answer)
        gpt3_tokens = tokenizer.tokenize(gpt3_answer)
        gpt4_tokens = tokenizer.tokenize(gpt4_answer)
        min_length = min(len(human_tokens), len(llama2_tokens), len(llama3_tokens), len(gpt3_tokens), len(gpt4_tokens))

        human_tokens = human_tokens[:min_length]
        llama2_tokens = llama2_tokens[:min_length]
        llama3_tokens = llama3_tokens[:min_length]
        gpt3_tokens = gpt3_tokens[:min_length]
        gpt4_tokens = gpt4_tokens[:min_length]

        adjusted_human_answers.append(tokenizer.convert_tokens_to_string(human_tokens))
        adjusted_llama2_answers.append(tokenizer.convert_tokens_to_string(llama2_tokens))
        adjusted_llama3_answers.append(tokenizer.convert_tokens_to_string(llama3_tokens))
        adjusted_gpt3_answers.append(tokenizer.convert_tokens_to_string(gpt3_tokens))
        adjusted_gpt4_answers.append(tokenizer.convert_tokens_to_string(gpt4_tokens))

    return adjusted_human_answers, adjusted_llama2_answers, adjusted_llama3_answers, adjusted_gpt3_answers, adjusted_gpt4_answers

# Function to calculate perplexities and save them
def calculate_and_save_perplexities(data, human_output_file, llama2_output_file, llama3_output_file, gpt3_output_file, gpt4_output_file):
    human_answers = [entry['human'] for entry in data['Answers']]
    llama2_answers = [entry['llama2'] for entry in data['Answers']]
    llama3_answers = [entry['llama3'] for entry in data['Answers']]
    gpt3_answers = [entry['chatGpt3'] for entry in data['Answers']]
    gpt4_answers = [entry['chatGpt4'] for entry in data['Answers']]

    # Load pre-trained model tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Adjust all answers to have the same token length
    human_answers,llama2_answers, llama3_answers, gpt3_answers, gpt4_answers = adjust_token_lengths(tokenizer, human_answers, llama2_answers, llama3_answers, gpt3_answers, gpt4_answers)

    # Calculate perplexities
    human_perplexities = [calculate_perplexity(answer) for answer in human_answers]
    llama2_perplexities = [calculate_perplexity(answer) for answer in llama2_answers]
    llama3_perplexities = [calculate_perplexity(answer) for answer in llama3_answers]
    gpt3_perplexities = [calculate_perplexity(answer) for answer in gpt3_answers]
    gpt4_perplexities = [calculate_perplexity(answer) for answer in gpt4_answers]

    # Ensure the 'pickles' directory exists
    if not os.path.exists('pickles'):
        os.makedirs('pickles')

    # Store the perplexities in pickle files
    with open(human_output_file, 'wb') as file:
        pickle.dump(human_perplexities, file)
    with open(llama2_output_file, 'wb') as file:
        pickle.dump(llama2_perplexities, file)
    with open(llama3_output_file, 'wb') as file:
        pickle.dump(llama3_perplexities, file)
    with open(gpt3_output_file, 'wb') as file:
        pickle.dump(gpt3_perplexities, file)
    with open(gpt4_output_file, 'wb') as file:
        pickle.dump(gpt4_perplexities, file)

# Check if pickle files exist and load from them if they do
def load_or_calculate_perplexities(data, human_output_file,llama2_output_file, llama3_output_file, gpt3_output_file, gpt4_output_file):
    if os.path.exists(human_output_file) and os.path.exists(llama2_output_file) and os.path.exists(llama3_output_file) and os.path.exists(gpt3_output_file) and os.path.exists(gpt4_output_file):
        with open(human_output_file, 'rb') as file:
            human_perplexities = pickle.load(file)
        with open(llama2_output_file, 'rb') as file:
            llama2_perplexities = pickle.load(file)
        with open(llama3_output_file, 'rb') as file:
            llama3_perplexities = pickle.load(file)
        with open(gpt3_output_file, 'rb') as file:
            gpt3_perplexities = pickle.load(file)
        with open(gpt4_output_file, 'rb') as file:
            gpt4_perplexities = pickle.load(file)
        print("Loaded perplexities from pickle files.")
    else:
        print("Calculating perplexities...")
        calculate_and_save_perplexities(data, human_output_file,llama2_output_file, llama3_output_file, gpt3_output_file, gpt4_output_file)
        with open(human_output_file, 'rb') as file:
            human_perplexities = pickle.load(file)
        with open(llama2_output_file, 'rb') as file:
            llama2_perplexities = pickle.load(file)
        with open(llama3_output_file, 'rb') as file:
            llama3_perplexities = pickle.load(file)
        with open(gpt3_output_file, 'rb') as file:
            gpt3_perplexities = pickle.load(file)
        with open(gpt4_output_file, 'rb') as file:
            gpt4_perplexities = pickle.load(file)

    return human_perplexities,llama2_perplexities, llama3_perplexities, gpt3_perplexities, gpt4_perplexities
=======
>>>>>>> 5078517ca55be9ffb26a577cc0607796bd630a23

# Example usage
#FILE_PATH = 'data_cleaning/prompt1_merged.json'
FILE_PATH = 'data_cleaning/prompt1_merged.json'  # Your specified JSON file path
HUMAN_OUTPUT_FILE = 'pickles/human_perplexities_prompt_1.pkl'
LLAMA2_OUTPUT_FILE = 'pickles/llama2_perplexities_prompt_1.pkl'
LLAMA3_OUTPUT_FILE = 'pickles/llama3_perplexities_prompt_1.pkl'
GPT3_OUTPUT_FILE = 'pickles/gpt3_perplexitiesprompt_1.pkl'
GPT4_OUTPUT_FILE = 'pickles/gpt4_perplexities_prompt_1.pkl'

<<<<<<< HEAD
# Read JSON data
data = read_json(FILE_PATH)
=======
perplexity1 = calculate_perplexity(text1)
perplexity2 = calculate_perplexity(text2)
perplexity3 = calculate_perplexity(text3)
perplexity4 = calculate_perplexity(text)
print(f'Perplexity1: {perplexity1}')
print(f'Perplexity2: {perplexity2}')
print(f'Perplexity3: {perplexity3}')
print(f'Perplexity: {perplexity4}')
>>>>>>> 5078517ca55be9ffb26a577cc0607796bd630a23

# Load or calculate perplexities
human_perplexities, llama2_perplexities, llama3_perplexities, gpt3_perplexities, gpt4_perplexities = load_or_calculate_perplexities(data, HUMAN_OUTPUT_FILE, LLAMA2_OUTPUT_FILE, LLAMA3_OUTPUT_FILE, GPT3_OUTPUT_FILE, GPT4_OUTPUT_FILE)

# Plot the perplexities
plot_perplexities(human_perplexities,llama2_perplexities, llama3_perplexities, gpt3_perplexities, gpt4_perplexities)
plot_KDE(human_perplexities,llama2_perplexities, llama3_perplexities, gpt3_perplexities, gpt4_perplexities)