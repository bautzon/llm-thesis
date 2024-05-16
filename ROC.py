import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import DecisionBoundaryDisplay
from DistanceMatrix import calculate_prompt1, calculate_prompt2, CalculationsObject


prompt1_human, prompt1_llama2_student, prompt1_llama3_student, prompt1_gpt3_student, prompt1_gpt3_plain, prompt1_gpt3_humanlike, prompt1_gpt4_student = calculate_prompt1()
prompt2_human, prompt2_llama2_student, prompt2_llama3_student, prompt2_gpt3_student, prompt2_gpt3_plain, prompt2_gpt3_humanlike, prompt2_gpt4_student = calculate_prompt2()

def logistic_regression_cos_var(human_cos_var, synthetic_cos_var):
    human_cosine_variance = np.array(human_cos_var)
    gpt4_cosine_variance = np.array(synthetic_cos_var)

    labels_array_human = np.zeros(human_cosine_variance.shape)
    labels_array_gpt4 = np.ones(gpt4_cosine_variance.shape)

    x = np.concatenate((human_cosine_variance, gpt4_cosine_variance)).reshape(-1, 1)
    y = np.concatenate((labels_array_human, labels_array_gpt4))


    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    # Choose a classifier (Support Vector Machine in this case)
    classifier = LogisticRegression()

    # Train the classifier
    classifier.fit(x_train, y_train)

    # Make predictions
    y_pred = classifier.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    
    # Plotting
    plt.figure(figsize=(15, 5))

    # Plot original data distributions
    plt.subplot(1, 3, 1)
    plt.hist(human_cosine_variance, bins=20, alpha=0.5, label='Array 1')
    plt.hist(gpt4_cosine_variance, bins=20, alpha=0.5, label='Array 2')
    plt.xlabel('Variance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Variances')
    plt.legend()

    # Plot decision boundary
    plt.subplot(1, 3, 2)
    DecisionBoundaryDisplay.from_estimator(classifier, x, response_method="predict", alpha=0.5)
    plt.scatter(x_train, y_train, c=y_train, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel('Variance')
    plt.ylabel('Class Label')
    plt.title('Decision Boundary')

    # Plot classifier's predictions
    plt.subplot(1, 3, 3)
    plt.scatter(x_test, y_test, color='blue', label='Actual')
    plt.scatter(x_test, y_pred, color='red', marker='x', label='Predicted')
    plt.xlabel('Variance')
    plt.ylabel('Class Label')
    plt.title('Actual vs Predicted Labels')
    plt.legend()

    plt.tight_layout()
    plt.show()

#logistic_regression_cos_var()

def logistic_regression_euclid_dist():
    human_distances, human_covariances = np.array(prompt1_human.distances), np.array(prompt1_human.covariances)
    human_2D = np.stack((human_distances, human_covariances),axis=1)
    
    robot_distances, robot_covariances = np.array(prompt1_gpt4_student.distances),np.array(prompt1_gpt4_student.covariances)
    
    robot_2D = np.stack((robot_distances, robot_covariances), axis=1)

    # Check and print the shape of individual arrays
    print("Shape of human_distances:", human_distances.shape)
    print("Shape of human_covariances:", human_covariances.shape)
    print("Shape of human_2D:", human_2D.shape)
    print("Shape of robot_distances:", robot_distances.shape)
    print("Shape of robot_covariances:", robot_covariances.shape)
    print("Shape of robot_2D:", robot_2D.shape)

    
    x = np.vstack((human_2D, robot_2D))
    print("Shape of x_2D:", x.shape)
    
    y_human = np.zeros(len(human_distances))
    y_robot = np.ones(len(robot_distances))
    y = np.concatenate([y_human,
                        y_robot])
    

    # y = np.concatenate((labels_array_human, labels_array_gpt4))
    print("Shape of y (combined labels):", y.shape)

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

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report: \n{report}")
    
    # # Plotting
    # plt.figure(figsize=(15, 5))

    # # Plot original data distributions
    # plt.subplot(1, 3, 1)
    # plt.hist(prompt1_human_euclid_distance, bins=20, alpha=0.5, label='Array 1')
    # plt.hist(prompt1_gpt4_euclid_distance, bins=20, alpha=0.5, label='Array 2')
    # plt.xlabel('Variance')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Variances')
    # plt.legend()

    # # Plot decision boundary
    # plt.subplot(1, 3, 2)
    # DecisionBoundaryDisplay.from_estimator(classifier, x, response_method="predict", alpha=0.5)
    # plt.scatter(x_train, y_train, c=y_train, edgecolors='k', cmap=plt.cm.coolwarm)
    # plt.xlabel('Variance')
    # plt.ylabel('Class Label')
    # plt.title('Decision Boundary')

    # # Plot classifier's predictions
    # plt.subplot(1, 3, 3)
    # plt.scatter(x_test, y_test, color='blue', label='Actual')
    # plt.scatter(x_test, y_pred, color='red', marker='x', label='Predicted')
    # plt.xlabel('Variance')
    # plt.ylabel('Class Label')
    # plt.title('Actual vs Predicted Labels')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
    
logistic_regression_cos_var(prompt2_human.cosine_variance_list,
                            prompt2_gpt4_student.cosine_variance_list )

#logistic_regression_euclid_dist()