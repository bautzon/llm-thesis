import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import DecisionBoundaryDisplay
from DistanceMatrix import get_openqa_calculations, get_eli5_calculations, get_prompt1_calculations, get_prompt2_calculations, CalculationsObject


prompt1_human, prompt1_llama2_student, prompt1_llama3_student, prompt1_gpt3_student, prompt1_gpt3_plain, prompt1_gpt3_humanlike, prompt1_gpt4_student = get_prompt1_calculations()
prompt2_human, prompt2_llama2_student, prompt2_llama3_student, prompt2_gpt3_student, prompt2_gpt3_plain, prompt2_gpt3_humanlike, prompt2_gpt4_student = get_prompt2_calculations()
eli5_human, eli5_llama2, eli5_llama3, eli5_chatGpt3 = get_eli5_calculations()
openqa_human, openqa_chatGpt3 = get_openqa_calculations()

def logistic_regression_cos_var(human_cos_var, synthetic_cos_var):
    human_cosine_variance = np.array(human_cos_var)
    gpt4_cosine_variance = np.array(synthetic_cos_var)

    labels_array_human = np.zeros(human_cosine_variance.shape)
    labels_array_gpt4 = np.ones(gpt4_cosine_variance.shape)
    
    x = np.concatenate((human_cosine_variance, gpt4_cosine_variance)).reshape(-1, 1)
    y = np.concatenate((labels_array_human, labels_array_gpt4))

    # # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # # Choose a classifier (Support Vector Machine in this case)
    classifier = LogisticRegression()

    # # Train the classifier
    classifier.fit(x_train, y_train)

    # # Make predictions
    y_pred = classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Variance of Cosine SimilarityAccuracy: {accuracy}")
    print(f"Classification Report: \n{report}")

    # Plotting
    plt.figure(figsize=(10, 5))

    # Plot original data distributions
    plt.subplot(1, 2, 1)
    plt.hist(human_cosine_variance, bins=20, alpha=0.5, label='Human Cosine Variance')
    plt.hist(gpt4_cosine_variance, bins=20, alpha=0.5, label='Synthetic Cosine Variance')
    plt.xlabel('Variance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Variances')
    plt.legend()



    # Plot classifier's predictions
    plt.subplot(1, 2, 2)
    plt.scatter(x_test, y_test, color='blue', label='Actual')
    plt.scatter(x_test, y_pred, color='red', marker='x', label='Predicted')
    plt.xlabel('Variance')
    plt.ylabel('Class Label')
    plt.title('Actual vs Predicted Labels')
    plt.legend()

    plt.tight_layout()
    plt.show()

def logistic_regression_euclid_dist(human_dist,human_cov, 
                                    synthetic_dist, synthetic_cov):
    
    human_distances, human_covariances = np.array(human_dist), np.array(human_cov)
    human_2D = np.stack((human_distances, human_covariances),axis=1)
    
    robot_distances, robot_covariances = np.array(synthetic_dist),np.array(synthetic_cov)
    robot_2D = np.stack((robot_distances, robot_covariances), axis=1)
    
    x = np.vstack((human_2D, robot_2D))
    
    y_human = np.zeros(len(human_distances))
    y_robot = np.ones(len(robot_distances))
    y = np.concatenate([y_human, y_robot])
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    #* Using stratify preserves the distribution across training and test sets, especially useful when dealing with imbalanced datasets

    # Choose a classifier (Logistic Regression)
    classifier = LogisticRegression()

    # Train the classifier
    classifier.fit(x_train, y_train)

    # Make predictions
    y_pred = classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy of Variance in Distance and Covariance: {accuracy}")
    print(f"Classification Report: \n{report}")
    
    # Plotting
    plt.figure(figsize=(15, 5))

    # Plot original data distributions
    plt.subplot(1, 3, 1)
    plt.hist(human_distances, bins=20, alpha=0.5, label='Human Distances')
    plt.hist(robot_distances, bins=20, alpha=0.5, label='Robot Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances')
    plt.legend()
    
     # Plot original data distributions
    plt.subplot(1, 3, 2)
    plt.hist(human_covariances, bins=20, alpha=0.5, label='Human Covariance')
    plt.hist(robot_covariances, bins=20, alpha=0.5, label='Robot Covariance')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances')
    plt.legend()
    

    # Plot decision boundary
    plt.subplot(1, 3, 3)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.coolwarm, marker='o', label='Training data')
    # Plot the testing point
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.coolwarm, marker='s', label='Testing data')
    plt.xlabel('Variance')
    plt.ylabel('Class Label')
    plt.title('Decision Boundary')
    DecisionBoundaryDisplay.from_estimator(
        classifier,
        x,
        response_method="predict",
        cmap=plt.cm.Spectral,
        alpha=0.8,
        xlabel="No. of Answers",
        ylabel="Average of Cosine Similarity of Answers"
    )
    plt.legend()
    
    
    plt.tight_layout()
    plt.show()


#Prompt1
def call_prompt1():
    logistic_regression_cos_var(prompt1_human.cosine_variance_per_answer, 
                                prompt1_gpt4_student.cosine_variance_per_answer)
    logistic_regression_euclid_dist(prompt1_human.distances, prompt1_human.covariances, prompt1_gpt4_student.distances, prompt1_gpt4_student.covariances)

#Prompt2
def call_prompt2():
    logistic_regression_cos_var(prompt2_human.cosine_variance_per_answer, 
                                prompt2_gpt4_student.cosine_variance_per_answer)
    logistic_regression_euclid_dist(prompt2_human.distances, prompt2_human.covariances, prompt2_gpt4_student.distances, prompt2_gpt4_student.covariances)

#ELI5
def call_eli5():
    logistic_regression_cos_var(eli5_human.cosine_variance_per_answer, 
                                eli5_llama3.cosine_variance_per_answer)
    logistic_regression_euclid_dist(eli5_human.distances, eli5_human.covariances, eli5_llama3.distances, eli5_llama3.covariances)

#OpenQA
def call_openqa():
    logistic_regression_cos_var(openqa_human.cosine_variance_per_answer, 
                                openqa_chatGpt3.cosine_variance_per_answer)
    logistic_regression_euclid_dist(openqa_human.distances, openqa_human.covariances, openqa_chatGpt3.distances, openqa_chatGpt3.covariances)

def main():
    call_prompt1()
    call_prompt2()
    #call_eli5()
    #call_openqa()
    
    
if __name__ == "__main__":
    main()