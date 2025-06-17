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

path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

print("Path to dataset files:", path)

file_path = os.path.join(path, "IMDB Dataset.csv")
df = pd.read_csv(file_path)

print(df.head())
reviews = df['review'].head(10)

for review in reviews:
    print(review)

test = reviews[0]


# Convert to lowercase
def text_lowercase(text):
    return text.lower()

test = text_lowercase(test)
print(' ')
print(f"lowercase : {test}")

# Remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

test = remove_punctuation(test)
print(' ')
print(f"remove punctuation : {test}")

# convert numbers into letters
p = inflect.engine()

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

test = convert_number(test)
print(' ')
print(f"convert numbers : {test}")

# remove stopwords
nltk.download('punkt_tab')
nltk.download('stopwords')


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text

test = remove_stopwords(test)
print(' ')
print(f"remove stopwords : {test}")

# stemming

stemmer = PorterStemmer()

def stem_words(text):
    stems = [stemmer.stem(word) for word in text]
    return stems

test = stem_words(test)
print(' ')
print(f"Stemming : {test}")



#Pipeline complet sur une seule critique
def preprocess_review(text):
    text = text_lowercase(text)
    text = remove_punctuation(text)
    text = convert_number(text)
    tokens = remove_stopwords(text)
    stems = stem_words(tokens)
    return ' '.join(stems)

# Application à tout le DataFrame
print("Prétraitement en cours...")
df_test = df

df_test["processed_review"] = df_test["review"].apply(preprocess_review)

# df_save = df["processed_review"]
print('*******')
print(df_test)




# Sauvegarde
output_dir= "data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "preprocessed.csv")
df_test['processed_review'].to_csv(output_path, index=False)

print(f"Prétraitement terminé. Fichier sauvegardé : {output_path}")


