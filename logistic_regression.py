import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import DecisionBoundaryDisplay
from DistanceMatrix import get_prompt1_calculations, get_prompt2_calculations, CalculationsObject


prompt1_human, prompt1_llama2_student, prompt1_llama3_student, prompt1_gpt3_student, prompt1_gpt3_plain, prompt1_gpt3_humanlike, prompt1_gpt4_student = get_prompt1_calculations()
prompt2_human, prompt2_llama2_student, prompt2_llama3_student, prompt2_gpt3_student, prompt2_gpt3_plain, prompt2_gpt3_humanlike, prompt2_gpt4_student = get_prompt2_calculations()

def logistic_regression_cos_var(human_cos_var, synthetic_cos_var):
    human_cosine_variance = np.array(human_cos_var)
    gpt4_cosine_variance = np.array(synthetic_cos_var)

    labels_array_human = np.zeros(human_cosine_variance.shape)
    labels_array_gpt4 = np.ones(gpt4_cosine_variance.shape)
    #* A literal fuckton of prints
    # Print the arrays and their shapes
    #print("human_cosine_variance:", human_cosine_variance)
    #print("Shape of human_cosine_variance:", human_cosine_variance.shape)
    #print("gpt4_cosine_variance:", gpt4_cosine_variance)
    #print("Shape of gpt4_cosine_variance:", gpt4_cosine_variance.shape)
    # Print the label arrays and their shapes
    #print("labels_array_human:", labels_array_human)
    #print("Shape of labels_array_human:", labels_array_human.shape)
    #print("labels_array_gpt4:", labels_array_gpt4)
    #print("Shape of labels_array_gpt4:", labels_array_gpt4.shape)

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
    plt.figure(figsize=(15, 5))

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

    #Check and print the shape of individual arrays
    # print("Shape of human_distances:", human_distances.shape)
    # print("Shape of human_covariances:", human_covariances.shape)
    # print("Shape of human_2D:", human_2D.shape)
    # print("Shape of robot_distances:", robot_distances.shape)
    # print("Shape of robot_covariances:", robot_covariances.shape)
    # print("Shape of robot_2D:", robot_2D.shape)

    
    x = np.vstack((human_2D, robot_2D))
    #print("Shape of x_2D:", x.shape)
    
    y_human = np.zeros(len(human_distances))
    y_robot = np.ones(len(robot_distances))
    y = np.concatenate([y_human, y_robot])
    

    #y = np.concatenate((labels_array_human, labels_array_gpt4))
    #print("Shape of y (combined labels):", y.shape)

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
    # plt.scatter(x[:, 0], x[:, 1], c=y, s=20, edgecolors="k")

    plt.tight_layout()
    
    
    plt.show()
    

logistic_regression_cos_var(prompt2_human.cosine_variance_per_answer, 
                            prompt2_gpt4_student.cosine_variance_per_answer)

logistic_regression_euclid_dist(prompt2_human.distances, prompt2_human.covariances, prompt2_gpt4_student.distances, prompt2_gpt4_student.covariances)