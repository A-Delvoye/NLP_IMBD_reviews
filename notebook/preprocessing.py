#region imports
import kagglehub
import pandas as pd
import os
import nltk
import string
import re
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

#region ntlk downloads
nltk.download('punkt')
nltk.download('stopwords')

#region Loading data from kagglehub...
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
print("Path to dataset files:", path)

file_path = os.path.join(path, "IMDB Dataset.csv")
df = pd.read_csv(file_path)

#region Global initializations
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
p = inflect.engine()

#region show Dataframe
# print(df.head())
# reviews = df['review'].head(10)

# for review in reviews:
#     print(review)

# test = reviews[0]

#region Functions

# 1. Convert to lowercase
def text_lowercase(text):
    return text.lower()

# 2. Remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# 3. Convert numbers into letters
def convert_number(text):
    temp_str = text.split()
    new_string = []

    for word in temp_str:
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)

        else:
            new_string.append(word)

    temp_str = ' '.join(new_string)
    return temp_str

# 4. remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text

# 5. Stemming
def stem_words(text):
    stems = [stemmer.stem(word) for word in text]
    return stems

#region Pipeline complète
def preprocess_review(text):
    text = text_lowercase(text)
    text = remove_punctuation(text)
    text = convert_number(text)
    tokens = remove_stopwords(text)
    stems = stem_words(tokens)
    return ' '.join(stems)


#__________________________________________________________________

# Application de la pipe
print("Prétraitement en cours...")
df_test = df.head(10)

df_test["processed_review"] = df_test["review"].apply(preprocess_review)

# Sauvegarde des données et export dans Dossier "data"
output_dir= "data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "preprocessed.csv")
df_test['processed_review'].to_csv(output_path, index=False)

print(f"Prétraitement terminé. Fichier sauvegardé : {output_path}")


#_______________________________________________________________

from sklearn.feature_extraction.text import CountVectorizer

# Initialiser le vectorizer
vectorizer = CountVectorizer()

# Appliquer sur les reviews prétraitées
X_bow = vectorizer.fit_transform(df_test["processed_review"])

# Optionnel : convertir en DataFrame lisible
bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())

# Afficher un extrait du BoW
print(bow_df.head())

