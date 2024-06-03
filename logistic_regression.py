import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import DecisionBoundaryDisplay
from DistanceMatrix import get_openqa_calculations, get_eli5_calculations, get_prompt1_calculations, get_prompt2_calculations, CalculationsObject

# Get the data
prompt1_human, prompt1_llama2_student, prompt1_llama3_student, prompt1_gpt3_student, prompt1_gpt3_plain, prompt1_gpt3_humanlike, prompt1_gpt4_student = get_prompt1_calculations()
prompt2_human, prompt2_llama2_student, prompt2_llama3_student, prompt2_gpt3_student, prompt2_gpt3_plain, prompt2_gpt3_humanlike, prompt2_gpt4_student = get_prompt2_calculations()
eli5_human, eli5_llama2, eli5_llama3, eli5_chatGpt3 = get_eli5_calculations()
openqa_human, openqa_chatGpt3 = get_openqa_calculations()

# Define logistic regression function
def logistic_regression_cos_var(train_human_cos_var, train_synthetic_cos_var, test_human_cos_var, test_synthetic_cos_var):
    train_human_cosine_variance = np.array(train_human_cos_var)
    train_synthetic_cosine_variance = np.array(train_synthetic_cos_var)

    test_human_cosine_variance = np.array(test_human_cos_var)
    test_synthetic_cosine_variance = np.array(test_synthetic_cos_var)

    train_labels_array_human = np.zeros(train_human_cosine_variance.shape)
    train_labels_array_synthetic = np.ones(train_synthetic_cosine_variance.shape)

    test_labels_array_human = np.zeros(test_human_cosine_variance.shape)
    test_labels_array_synthetic = np.ones(test_synthetic_cosine_variance.shape)

    x_train = np.concatenate((train_human_cosine_variance, train_synthetic_cosine_variance)).reshape(-1, 1)
    y_train = np.concatenate((train_labels_array_human, train_labels_array_synthetic))

    x_test = np.concatenate((test_human_cosine_variance, test_synthetic_cosine_variance)).reshape(-1, 1)
    y_test = np.concatenate((test_labels_array_human, test_labels_array_synthetic))

    # Split the data
    
    # Choose a classifier (Logistic Regression in this case)
    classifier = LogisticRegression()

    # Train the classifier
    classifier.fit(x_train, y_train)

    # Make predictions
    y_pred = classifier.predict(x_test)
    y_prob = classifier.predict_proba(x_test)[:, 1]

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Variance of Cosine Similarity Accuracy: {accuracy}")
    print(f"Classification Report: \n{report}")

    # Plotting
    plt.figure(figsize=(12, 5))

    # Plot original data distributions
    plt.subplot(1, 2, 1)
    plt.hist(train_human_cosine_variance, bins=20, alpha=0.5, label='Train Human Cosine Variance')
    plt.hist(train_synthetic_cosine_variance, bins=20, alpha=0.5, label='Train Synthetic Cosine Variance')
    plt.hist(test_human_cosine_variance, bins=20, alpha=0.5, label='Test Human Cosine Variance', histtype='step')
    plt.hist(test_synthetic_cosine_variance, bins=20, alpha=0.5, label='Test Synthetic Cosine Variance', histtype='step')
    plt.xlabel('Variance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Variances')
    plt.legend()

    # Plot classifier's predictions with sigmoid curve
    plt.subplot(1, 2, 2)
    plt.scatter(x_test, y_test, color='blue', label='Actual')
    plt.scatter(x_test, y_pred, color='red', marker='x', label='Predicted')

    # Add sigmoid curve
    x_values = np.linspace(x_train.min(), x_train.max(), 300)
    y_values = classifier.predict_proba(x_values.reshape(-1, 1))[:, 1]  # Get the probability of class 1
    plt.plot(x_values, y_values, color='green', linewidth=2, label='Sigmoid Curve')

    plt.xlabel('Variance')
    plt.ylabel('Class Label')
    plt.title('Actual vs Predicted Labels with Sigmoid Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Define functions to call logistic regression with specific datasets
def call_prompt1():
    logistic_regression_cos_var(prompt1_human.mean_cosine_variance, prompt1_gpt4_student.mean_cosine_variance, prompt1_human.mean_cosine_variance, prompt1_gpt4_student.mean_cosine_variance)

def call_prompt2():
    logistic_regression_cos_var(prompt2_human.mean_cosine_variance, prompt2_gpt4_student.mean_cosine_variance, prompt2_human.mean_cosine_variance, prompt2_gpt4_student.mean_cosine_variance)

def call_eli5():
    logistic_regression_cos_var(eli5_human.mean_cosine_variance, eli5_llama3.mean_cosine_variance, eli5_human.mean_cosine_variance, eli5_llama3.mean_cosine_variance)

def call_openqa():
    logistic_regression_cos_var(openqa_human.mean_cosine_variance, openqa_chatGpt3.mean_cosine_variance, openqa_human.mean_cosine_variance, openqa_chatGpt3.mean_cosine_variance)


# Functions for different train-test scenarios
def train_on_prompt1_test_on_prompt1():
    logistic_regression_cos_var(
        prompt1_human.mean_cosine_variance,
        prompt1_gpt4_student.mean_cosine_variance,
        prompt1_human.mean_cosine_variance,
        prompt1_gpt4_student.mean_cosine_variance
    )

def train_on_prompt1_test_on_prompt2():
    logistic_regression_cos_var(
        prompt1_human.mean_cosine_variance, 
        prompt1_gpt4_student.mean_cosine_variance,
        prompt2_human.mean_cosine_variance,
        prompt2_gpt4_student.mean_cosine_variance
    )


def train_on_prompt2_test_on_prompt2():
    logistic_regression_cos_var(
        prompt2_human.mean_cosine_variance, 
        prompt2_gpt4_student.mean_cosine_variance,
        prompt2_human.mean_cosine_variance,
        prompt2_gpt4_student.mean_cosine_variance
    )


def train_on_prompt2_test_on_prompt1():
    logistic_regression_cos_var(
        prompt2_human.mean_cosine_variance, 
        prompt2_gpt4_student.mean_cosine_variance,
        prompt1_human.mean_cosine_variance,
        prompt1_gpt4_student.mean_cosine_variance
    )

# Main function to run the chosen analysis
def main():
    # Uncomment the function call you want to test
    #train_on_prompt1_test_on_prompt1()
    #train_on_prompt1_test_on_prompt2()
    # train_on_prompt2_test_on_prompt2()
    # train_on_prompt2_test_on_prompt1()
    # train_on_eli5_test_on_eli5()
    # train_on_eli5_test_on_prompt1()
    # train_on_eli5_test_on_prompt2()
    # train_on_openqa_test_on_openqa()
    # train_on_openqa_test_on_prompt1()
    # train_on_openqa_test_on_prompt2()
    # train_on_openqa_test_on_eli5()
    train_on_prompt1_test_on_eli5()

if __name__ == "__main__":
    main()
