import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Step 1: Read the JSON File
file_path = 'Test-Data/combined.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Ensure data contains "Answers" key
if "Answers" not in data:
    raise ValueError("Expected the JSON data to contain an 'Answers' key")

answers_data = data["Answers"]

# Print the type and content of the loaded data for debugging
#print(f"Type of data: {type(answers_data)}")
#print("Content of the JSON file:")
#print(json.dumps(answers_data[:5], indent=2))  # Print first 5 items for inspection

# Step 2: Data Preprocessing
# Extract answers and labels
answers = []
labels = []
for item in answers_data:
    try:
        answers.append(item['answer'])
        labels.append(1 if item['creator'] == 'ai' else 0)
    except KeyError as e:
        print(f"Missing key in item: {item}, error: {e}")

# Debugging: Print the first few answers and labels to verify extraction
#print("Sample answers:", answers[:5])
#print("Sample labels:", labels[:5])

# Step 3: Split Data
x_train, x_test, y_train, y_test = train_test_split(answers, labels, test_size=0.8, random_state=69)

# Step 4: Vectorize the sentences using TF-IDF and Build the RBF Classifier Pipeline
vectorizer = TfidfVectorizer()
classifier = SVC(kernel='rbf')
clf = make_pipeline(vectorizer, classifier)

# Step 5: Train the classifier
clf.fit(x_train, y_train)

# Step 6: Evaluate the model
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))

