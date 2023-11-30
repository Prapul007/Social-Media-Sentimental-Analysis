import nltk
import re
import unicodedata
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Reading the dataset
df = pd.read_csv("train.csv", encoding='latin1')

# Removing the unwanted data
df.drop(df[df['sentiment'] == "neutral"].index, inplace=True)

# Downloading the stopwords
nltk.download('stopwords')

# Instantiate tokenizer, stemmer,stopwords, and WordNetLemmatizer
tokenizer = RegexpTokenizer(r'\w+')
ps = PorterStemmer()
en_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Function for cleaning text
def getCleanedText(text):
    clean_text = str(text)

    # Remove URLs
    clean_text = re.sub(r'http\S+|www.\S+|mailto:\S+', '', clean_text)

    # Remove hashtags
    clean_text = re.sub(r'#\w+', '', clean_text)

    # Remove mentions
    clean_text = re.sub(r'@\w+', '', clean_text)

    # Remove non-alphabetic characters
    clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)

    # Remove extra spaces
    clean_text = ' '.join(clean_text.split())

    # Remove diacritics (accents)
    clean_text = ''.join(c for c in unicodedata.normalize('NFD', clean_text) if
                         unicodedata.category(c) != 'Mn')

    # Convert to lowercase
    clean_text = clean_text.lower()

    # Tokenize the cleaned text
    tokens = tokenizer.tokenize(clean_text)

    # Remove stopwords
    new_tokens = [token for token in tokens if token not in en_stopwords]

    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in new_tokens]

    # Join tokens back into a cleaned text
    clean_text = " ".join(lemmatized_tokens)

    return clean_text


# Creating a new column of cleaned data
df['new'] = df['text'].apply(getCleanedText)


# Assigning the columns to x and y
x = df['new']
y = df['sentiment']


# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # You can adjust max_features as needed
x_train_tfidf = tfidf_vectorizer.fit_transform(x)


# Train the Naive Bayes Classifier
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train_tfidf, y)



# Make predictions on the given input from user
def prediction(input_txt):
    # It Checks the is with spaces or with a normal sentence
    if input_txt.isspace():
        sentiment = [0]
        return sentiment
    else:
        test = [input_txt]
        x_test = getCleanedText(text = test)
        x_test_tfidf = tfidf_vectorizer.transform([x_test])
        sentiment = classifier.predict(x_test_tfidf)
        return sentiment



