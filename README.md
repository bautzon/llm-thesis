# Platt

---

This is the official repository for the Platt project, a project created for a Master's Thesis in Computer Science at the IT University of Copenhagen.

The repository proposes a method for detecting AI-generated text.

**Disclaimer**: Does only work on english texts.

## Usage

### Installation

The tool was dveloped using Python 3.12

Install requirements
```bash
git clone https://github.com/bautzon/llm-thesis.git
cd llm-thesis
pip install -r requirements.txt
```

### Models

Some files are too large to be stored in the repository. This includes the models that can be used for generating calculations.
We propose using the link here to download the models and place them in the `models` directory.
- [GoogleNews-vectors-negative300.bin](https://code.google.com/archive/p/word2vec/)
- [crawl-300d-2M.vec](https://fasttext.cc/docs/en/english-vectors.html)
- [wiki-news-300d-1M.vec](https://fasttext.cc/docs/en/english-vectors.html)

### Running the tool

Notice the flag `GENERATE_NEW_DATA` at line 11. If this is set to true the program will generate data again. This is a time-consuming process and should only be done if you want to generate new data.
When set to false it will reuse the data in the pickle files.

### Demo

A demo can be found in the `demo.py`file
    
```bash
python demo.py
```

## Limitations

No AI detectors are perfect and this tool is no exception. The tool is based on the assumption that AI-generated text is different from human-generated text. This is not always the case as AI evolves and the tool might not be able to detect AI-generated text in all cases.
It shouls also be noted that it works on english texts only.

## Project structure

The project is structured as follows:

- `data_processing/`: Contains various functions used for generating, processing and cleaning data.
  - `extracted_answers/` : contains all answers from the datasets in .txt-files
  - `functions`: contains javascript files used for cleaning, processing and generating the data
  - `ai output`: 
- `models/`: Contains the models used for generating calculations
- `pickles/`: Contains the pickle files used for storing data
- `Test-Data/` : Contains the test data used for the project
- `DistanceMatrix.py`: 
- `LagrangeInterpolation.py`:
- `lda.py`:
- `logistic_regression.py`
- `nucleus_sampling.py`: Contains the NucleausSampling class used for generating the Nucleaus Sampling model
- `pca.py`: Contains the PCA class used for generating the PCA model
- `perplexity`:
- `RungeExample.py`:
- `sampling.py`:
- `simple_classifier`: 


### Generating AI data

The repository still contains the javascript file `chatGPT-generator.js` which is only usable if you have a key for the OpenAI API. This key is not provided in the repository.
