import pandas as pd #might require an install

#File path
file_path = "asap-aes/training_set_rel3.tsv"

#Read the csv file into a dataframe
df = pd.read_csv(file_path, delimiter=',', usecols=[0, 1, 2], skiprows=5, skip_blank_lines=True,  encoding="ISO-8859-1")

print(df.head())