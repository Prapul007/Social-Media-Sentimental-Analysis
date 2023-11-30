## Sentiment Analysis for Social Media Sentence

This project involves sentiment analysis for social media sentence. It uses an SVC (Support Vector Classifier) model trained on cleaned and preprocessed data from a dataset.

### Objective

The primary objective of this project is to predict sentiment i.e positive or negative from text data by employing natural language processing (NLP) techniques.

### Dataset

The dataset used for training and testing the model is sourced from 'train.csv' and 'test.csv'. It is preprocessed to remove neutral sentiment entries and many other things like links, extra spaces, mentions etc, to clean text for analysis.
You can access the dataset from Kaggle by following this link: [Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset).


### Methodology

The preprocessing pipeline includes:
- Removal of URLs, hashtags, mentions, and non-alphabetic characters
- Lowercasing the text
- Tokenization and removal of stopwords
- Lemmatization of tokens
- TF-IDF vectorization of the cleaned text for modeling

### Model

The SVC model with an RBF (Radial Basis Function) kernel is utilized for sentiment prediction. It achieves an accuracy of 89.03% on the test data, To verify accuracy, you can refer to the "accuracy.ipynb" file within this repository.

### Frontend

The application is built using Streamlit, providing a simple user interface:
- Users can enter social media statements in the text area provided.
- On clicking the "Analyze" button, the sentiment analysis is performed using the SVC model.
- If the input is not empty, the predicted sentiment (positive or negative) is displayed using emoji icons.
- In case of an error or no sentiment prediction, an appropriate message is displayed.


