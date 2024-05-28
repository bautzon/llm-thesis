import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_perplexity(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def pad_or_truncate_answers(answers, max_length):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    adjusted_answers = []
    for answer in answers:
        tokens = tokenizer.tokenize(answer)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens.extend(['<pad>'] * (max_length - len(tokens)))
        adjusted_answers.append(tokenizer.convert_tokens_to_string(tokens))
    return adjusted_answers

def calculate_perplexities_per_position(answers, model, tokenizer):
    max_length = max(len(tokenizer.tokenize(answer)) for answer in answers)
    answers = pad_or_truncate_answers(answers, max_length)
    
    perplexities = np.zeros((len(answers), max_length))
    
    for i, answer in enumerate(answers):
        tokens = tokenizer.tokenize(answer)
        for j in range(min(len(tokens), max_length)):
            partial_answer = tokenizer.convert_tokens_to_string(tokens[:j+1])
            perplexities[i, j] = calculate_perplexity(partial_answer, model, tokenizer)
    
    return perplexities

def plot_kde_3d(perplexities):
    x = np.arange(perplexities.shape[1])
    y = np.arange(perplexities.shape[0])
    x, y = np.meshgrid(x, y)
    z = perplexities

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')

    ax.set_title('3D KDE of Perplexities')
    ax.set_xlabel('Word Position')
    ax.set_ylabel('Answer Index')
    ax.set_zlabel('Perplexity')

    plt.show()

def calculate_and_save_perplexities(data, human_output_file, llama2_output_file, llama3_output_file, gpt3_output_file, gpt4_output_file):
    human_answers = [entry['human'] for entry in data['Answers']]
    llama2_answers = [entry['llama2'] for entry in data['Answers']]
    llama3_answers = [entry['llama3'] for entry in data['Answers']]
    gpt3_answers = [entry['chatGpt3'] for entry in data['Answers']]
    gpt4_answers = [entry['chatGpt4'] for entry in data['Answers']]

    # Load pre-trained model tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Load pre-trained model (weights)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # Calculate perplexities
    human_perplexities = calculate_perplexities_per_position(human_answers, model, tokenizer)
    llama2_perplexities = calculate_perplexities_per_position(llama2_answers, model, tokenizer)
    llama3_perplexities = calculate_perplexities_per_position(llama3_answers, model, tokenizer)
    gpt3_perplexities = calculate_perplexities_per_position(gpt3_answers, model, tokenizer)
    gpt4_perplexities = calculate_perplexities_per_position(gpt4_answers, model, tokenizer)

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
def load_or_calculate_perplexities(data, human_output_file, llama2_output_file, llama3_output_file, gpt3_output_file, gpt4_output_file):
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
        calculate_and_save_perplexities(data, human_output_file, llama2_output_file, llama3_output_file, gpt3_output_file, gpt4_output_file)
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

    return human_perplexities, llama2_perplexities, llama3_perplexities, gpt3_perplexities, gpt4_perplexities

def main():
    # Change these file paths to run on a different dataset
    FILE_PATH = 'data_cleaning/prompt1_merged.json'
    HUMAN_OUTPUT_FILE = 'pickles/human_perplexities_prompt_1_3d.pkl'
    LLAMA2_OUTPUT_FILE = 'pickles/llama2_perplexities_prompt_1_3d.pkl'
    LLAMA3_OUTPUT_FILE = 'pickles/llama3_perplexities_prompt_1_3d.pkl'
    GPT3_OUTPUT_FILE = 'pickles/gpt3_perplexities_prompt_1_3d.pkl'
    GPT4_OUTPUT_FILE = 'pickles/gpt4_perplexities_prompt_1_3d.pkl'

    # Read JSON data
    data = read_json(FILE_PATH)

    # Load or calculate perplexities
    human_perplexities, llama2_perplexities, llama3_perplexities, gpt3_perplexities, gpt4_perplexities = load_or_calculate_perplexities(data, HUMAN_OUTPUT_FILE, LLAMA2_OUTPUT_FILE, LLAMA3_OUTPUT_FILE, GPT3_OUTPUT_FILE, GPT4_OUTPUT_FILE)

    # Plot the 3D KDE of perplexities
    plot_kde_3d(human_perplexities)  # Adjust to plot different sets of perplexities if needed

if __name__ == '__main__':
    main()