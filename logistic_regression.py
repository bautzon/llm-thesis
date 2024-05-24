import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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

    labels_array_train_human = np.zeros(train_human_cosine_variance.shape)
    labels_array_train_synthetic = np.ones(train_synthetic_cosine_variance.shape)

    labels_array_test_human = np.zeros(test_human_cosine_variance.shape)
    labels_array_test_synthetic = np.ones(test_synthetic_cosine_variance.shape)

    x_train = np.concatenate((train_human_cosine_variance, train_synthetic_cosine_variance)).reshape(-1, 1)
    y_train = np.concatenate((labels_array_train_human, labels_array_train_synthetic))

    x_test = np.concatenate((test_human_cosine_variance, test_synthetic_cosine_variance)).reshape(-1, 1)
    y_test = np.concatenate((labels_array_test_human, labels_array_test_synthetic))

    # Choose a classifier (Logistic Regression in this case)
    classifier = LogisticRegression()

    # Train the classifier
    classifier.fit(x_train, y_train)

    # Make predictions
    y_pred = classifier.predict(x_test)

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
    plt.hist(test_human_cosine_variance, bins=20, alpha=0.5, label='Test Human Cosine Variance', linestyle='dashed')
    plt.hist(test_synthetic_cosine_variance, bins=20, alpha=0.5, label='Test Synthetic Cosine Variance', linestyle='dashed')
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

# Functions for different train-test scenarios
def train_on_prompt1_test_on_prompt1():
    logistic_regression_cos_var(
        prompt1_human.cosine_variance_per_answer,
        prompt1_gpt4_student.cosine_variance_per_answer,
        prompt1_human.cosine_variance_per_answer,
        prompt1_gpt4_student.cosine_variance_per_answer
    )

def train_on_prompt1_test_on_prompt2():
    logistic_regression_cos_var(
        prompt1_human.cosine_variance_per_answer, 
        prompt1_gpt4_student.cosine_variance_per_answer,
        prompt2_human.cosine_variance_per_answer,
        prompt2_gpt4_student.cosine_variance_per_answer
    )

def train_on_prompt2_test_on_prompt2():
    logistic_regression_cos_var(
        prompt2_human.cosine_variance_per_answer, 
        prompt2_gpt4_student.cosine_variance_per_answer,
        prompt2_human.cosine_variance_per_answer,
        prompt2_gpt4_student.cosine_variance_per_answer
    )

def train_on_prompt2_test_on_prompt1():
    logistic_regression_cos_var(
        prompt2_human.cosine_variance_per_answer, 
        prompt2_gpt4_student.cosine_variance_per_answer,
        prompt1_human.cosine_variance_per_answer,
        prompt1_gpt4_student.cosine_variance_per_answer
    )

def train_on_eli5_test_on_eli5():
    logistic_regression_cos_var(
        eli5_human.cosine_variance_per_answer,
        eli5_llama3.cosine_variance_per_answer,
        eli5_human.cosine_variance_per_answer,
        eli5_llama3.cosine_variance_per_answer
    )

def train_on_eli5_test_on_prompt1():
    logistic_regression_cos_var(
        eli5_human.cosine_variance_per_answer,
        eli5_llama3.cosine_variance_per_answer,
        prompt1_human.cosine_variance_per_answer,
        prompt1_gpt4_student.cosine_variance_per_answer
    )

def train_on_eli5_test_on_prompt2():
    logistic_regression_cos_var(
        eli5_human.cosine_variance_per_answer,
        eli5_llama3.cosine_variance_per_answer,
        prompt2_human.cosine_variance_per_answer,
        prompt2_gpt4_student.cosine_variance_per_answer
    )

def train_on_openqa_test_on_openqa():
    logistic_regression_cos_var(
        openqa_human.cosine_variance_per_answer,
        openqa_chatGpt3.cosine_variance_per_answer,
        openqa_human.cosine_variance_per_answer,
        openqa_chatGpt3.cosine_variance_per_answer
    )

def train_on_openqa_test_on_prompt1():
    logistic_regression_cos_var(
        openqa_human.cosine_variance_per_answer,
        openqa_chatGpt3.cosine_variance_per_answer,
        prompt1_human.cosine_variance_per_answer,
        prompt1_gpt4_student.cosine_variance_per_answer
    )

def train_on_openqa_test_on_prompt2():
    logistic_regression_cos_var(
        openqa_human.cosine_variance_per_answer,
        openqa_chatGpt3.cosine_variance_per_answer,
        prompt2_human.cosine_variance_per_answer,
        prompt2_gpt4_student.cosine_variance_per_answer
    )

def train_on_openqa_test_on_eli5():
    logistic_regression_cos_var(
        openqa_human.cosine_variance_per_answer,
        openqa_chatGpt3.cosine_variance_per_answer,
        eli5_human.cosine_variance_per_answer,
        eli5_llama3.cosine_variance_per_answer
    )
def train_on_prompt1_test_on_eli5():
    logistic_regression_cos_var(
        prompt1_human.cosine_variance_per_answer, 
        prompt1_gpt4_student.cosine_variance_per_answer,
        eli5_human.cosine_variance_per_answer,
        eli5_llama3.cosine_variance_per_answer
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