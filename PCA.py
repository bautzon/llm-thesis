import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from DistanceMatrix import get_openqa_calculations, get_eli5_calculations, get_prompt1_calculations, get_prompt2_calculations
import pandas as pd


MODEL = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
# MODEL = KeyedVectors.load_word2vec_format("models/crawl-300d-2M.vec")
# MODEL = KeyedVectors.load_word2vec_format("models/wiki-news-300d-1M.vec")


prompt1_human, prompt1_llama2_student, prompt1_llama3_student, prompt1_gpt3_student, prompt1_gpt3_plain, prompt1_gpt3_humanlike, prompt1_gpt4_student = get_prompt1_calculations()
prompt2_human, prompt2_llama2_student, prompt2_llama3_student, prompt2_gpt3_student, prompt2_gpt3_plain, prompt2_gpt3_humanlike, prompt2_gpt4_student = get_prompt2_calculations()
eli5_human, eli5_llama2, eli5_llama3, eli5_chatGpt3 = get_eli5_calculations()
openqa_human, openqa_chatGpt3 = get_openqa_calculations()

human_creator = ['human'] * len(prompt1_human.distances)
ai_creator = ['ai'] * len(prompt1_llama2_student.distances)

llama3_dataFrame = pd.DataFrame({
    "distance": prompt1_llama3_student.distances + prompt1_human.distances,
    "covariance": prompt1_llama3_student.covariances + prompt1_human.covariances,
    "cosine variance": prompt1_llama3_student.cosine_variance_per_answer + prompt1_human.cosine_variance_per_answer,
    "creator": ai_creator + human_creator
})

features = llama3_dataFrame[['distance', 'covariance', 'cosine variance']]

target = llama3_dataFrame['creator']

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

pca = PCA(n_components=3)

principalComponents = pca.fit_transform(features_standardized)

principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2', 'principal component 3'])

finalDf = pd.concat([principalDf, target], axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_zlabel('Principal Component 3', fontsize=15)
ax.set_title('3 component PCA', fontsize=20)

targets = ['human', 'ai']

colors = ['blue', 'g']

for target, color in zip(targets, colors):
    indicesToKeep = finalDf['creator'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
ax.legend(targets)
ax.grid()

plt.show()
