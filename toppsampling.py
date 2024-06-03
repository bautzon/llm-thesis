import pickle
import os
# Ensure the 'pickles' directory exists
if not os.path.exists('pickles'):
    os.makedirs('pickles')

with open('pickles/human_probs.pkl', 'rb') as file:
    obj = pickle.load(file)
print(obj)