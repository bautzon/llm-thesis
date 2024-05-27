import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from DistanceMatrix import get_openqa_calculations, get_eli5_calculations, get_prompt1_calculations, get_prompt2_calculations, CalculationsObject
import pandas as pd

prompt1_human, prompt1_llama2_student, prompt1_llama3_student, prompt1_gpt3_student, prompt1_gpt3_plain, prompt1_gpt3_humanlike, prompt1_gpt4_student = get_prompt1_calculations()
prompt2_human, prompt2_llama2_student, prompt2_llama3_student, prompt2_gpt3_student, prompt2_gpt3_plain, prompt2_gpt3_humanlike, prompt2_gpt4_student = get_prompt2_calculations()
# eli5_human, eli5_llama2, eli5_llama3, eli5_chatGpt3, eli5_chatGpt4 = get_eli5_calculations()
# openqa_human, openqa_llama2, openqa_llama3, openqa_chatGpt3, openqa_chatGpt4 = get_openqa_calculations()


def calculate_metrics(*students):
    distances = [distance for student in students for distance in student.mean_distances]
    covariances = [covariance for student in students for covariance in student.mean_covariances]
    cosine_variance = [cosine_var for student in students for cosine_var in student.mean_cosine_variance]
    return distances, covariances, cosine_variance


prompt1_ai_distances, prompt1_ai_covariances, prompt1_ai_cosine_variance = calculate_metrics(prompt1_llama3_student, prompt1_llama2_student, prompt1_gpt3_student, prompt1_gpt4_student)
prompt2_ai_distances, prompt2_ai_covariances, prompt2_ai_cosine_variance = calculate_metrics(prompt2_llama3_student, prompt2_llama2_student, prompt2_gpt3_student, prompt2_gpt4_student)
# eli5_distances, eli5_covariances, eli5_cosine_variance = calculate_metrics(eli5_llama2, eli5_llama3, eli5_chatGpt3, eli5_chatGpt4)
# openqa_distances, openqa_covariances, openqa_cosine_variance = calculate_metrics(openqa_llama2, openqa_llama3, openqa_chatGpt3, openqa_chatGpt4)

ai_creator = ['ai'] * 800
human_creator = ['human'] * 200

essay_dataFrame = pd.DataFrame({
     "distance": prompt1_ai_distances + prompt2_ai_distances + prompt1_human.mean_distances + prompt2_human.mean_distances,
     "covariance":prompt1_ai_covariances + prompt2_ai_covariances + prompt1_human.mean_covariances + prompt2_human.mean_covariances,
     "cosine variance": prompt1_ai_cosine_variance + prompt2_ai_cosine_variance + prompt1_human.mean_cosine_variance + prompt2_human.mean_cosine_variance,
     "creator": ai_creator + human_creator
})

# essay_dataFrame = pd.DataFrame({
#     "distance": openqa_distances + openqa_human.distances,
#     "covariance": openqa_covariances + openqa_human.covariances,
#     "cosine variance": openqa_cosine_variance + openqa_human.mean_cosine_variance,
#     "creator": ['ai'] * 400 + ['human'] * 100
# })

# essay_dataFrame = pd.DataFrame({
#     "distance": eli5_distances + eli5_human.distances,
#     "covariance": eli5_covariances + eli5_human.covariances,
#     "cosine variance": eli5_cosine_variance + eli5_human.cosine_variance_per_answer,
#     "creator": ['ai'] * 400 + ['human'] * 100
# })

features = essay_dataFrame[['distance', 'covariance', 'cosine variance']]

target = essay_dataFrame['creator']

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(features_standardized)

principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2', 'principal component 3'])

finalDf = pd.concat([principalDf, target], axis=1)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

fig = plt.figure(2, figsize=(8, 8))
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
