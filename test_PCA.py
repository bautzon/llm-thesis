import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from scipy.spatial.distance import cosine
#Working
# Initialize the model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Sample texts
human_text = "Triumph Engineering Co Ltd, a defunct British motorcycle manufacturerTriumph Motorcycles Ltd , a current British motorcycle manufacturer"
ai_text = "Triumph Motorcycles Ltd, a British company, makes Triumph motorcycles. The company was originally established in 1984 by John Bloor after the original company, Triumph Engineering, went into receivership. They are the largest British motorcycle manufacturer."

# Function to compute perplexity
def compute_perplexity(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item()

# Function to compute features
def compute_features(text):
    words = text.split()
    sentences = [' '.join(words[:i+1]) for i in range(len(words))]
    distances = []
    covariances = []
    cosine_similarities = []
    perplexities = []
    intermediate_embeddings = []

    for i, sentence in enumerate(sentences):
        inputs = tokenizer(sentence, return_tensors='pt')
        outputs = model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[-1].squeeze(0).mean(dim=0).detach()
        
        if i > 0:
            intermediate_mean = torch.stack(intermediate_embeddings).mean(dim=0).detach()
            cos_sim = 1 - cosine(embedding.cpu().numpy(), intermediate_mean.cpu().numpy())
        else:
            cos_sim = 1.0  # Self-similarity

        perplexity = compute_perplexity(sentence)
        
        # Dummy values for distances and covariances since they are not defined in the context
        distance = 0.0
        covariance = 0.0
        
        distances.append(distance)
        covariances.append(covariance)
        cosine_similarities.append(cos_sim)
        perplexities.append(perplexity if not np.isnan(perplexity) else 0.0)
        intermediate_embeddings.append(embedding)

    return distances, covariances, cosine_similarities, perplexities

# Function to update animation
def update(num, human_features, ai_features, ax, colors):
    ax.clear()
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('PCA of Text Features Over Time (Human vs AI)', fontsize=20)
    
    for label, color in zip(['human', 'ai'], colors):
        if label == 'human':
            features = human_features[:num+1]
        else:
            features = ai_features[:num+1]

        if len(features) > 1:
            pca = PCA(n_components=2)
            scaler = StandardScaler()
            features_standardized = scaler.fit_transform(features)
            principalComponents = pca.fit_transform(features_standardized)
            principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
            
            ax.plot(principalDf['principal component 1'], principalDf['principal component 2'], color=color, marker='o', linestyle='-')
        else:
            ax.scatter(0, 0, color=color, marker='o')  # Plot the first point at origin
        
    ax.legend(['human', 'ai'])
    ax.grid()

# Main processing
def process_text(human_text, ai_text):
    human_distances, human_covariances, human_cosine_similarities, human_perplexities = compute_features(human_text)
    ai_distances, ai_covariances, ai_cosine_similarities, ai_perplexities = compute_features(ai_text)

    human_features = pd.DataFrame({
        'distance': human_distances,
        'covariance': human_covariances,
        'cosine similarity': human_cosine_similarities,
        'perplexity': human_perplexities
    })

    ai_features = pd.DataFrame({
        'distance': ai_distances,
        'covariance': ai_covariances,
        'cosine similarity': ai_cosine_similarities,
        'perplexity': ai_perplexities
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue', 'red']
    
    ani = FuncAnimation(fig, update, frames=len(human_features), fargs=(human_features, ai_features, ax, colors), repeat=False)
    
    # Save the animation or display it
    ani.save('pca_animation.gif', writer='imagemagick')
    plt.show()

# Process and plot for human and AI text
process_text(human_text, ai_text)