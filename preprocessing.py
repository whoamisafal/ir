import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

BASE_DIR = os.path.expanduser("~")
INPUT_FOLDER = os.path.join(BASE_DIR, "Desktop", "all_articles")
TOKENIZED_FOLDER = os.path.join(BASE_DIR, "Desktop", "tokenized_articles")
CLEAN_FOLDER = os.path.join(BASE_DIR, "Desktop", "clean_articles")

os.makedirs(TOKENIZED_FOLDER, exist_ok=True)
os.makedirs(CLEAN_FOLDER, exist_ok=True)

# -----------------------------
# Tokenize
# -----------------------------
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".txt"):
        with open(os.path.join(INPUT_FOLDER, filename), "r", encoding="utf-8") as f:
            text = f.read()

        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'\b(?:ul|li)\b', '', text)

        # Tokenize
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        with open(os.path.join(TOKENIZED_FOLDER, filename), "w", encoding="utf-8") as f:
            f.write(" ".join(tokens))

# -----------------------------
# Stopword removal + lemmatization
# -----------------------------
for filename in os.listdir(TOKENIZED_FOLDER):
    if filename.endswith(".txt"):
        with open(os.path.join(TOKENIZED_FOLDER, filename), "r", encoding="utf-8") as f:
            tokens = f.read().split()

        filtered = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]

        with open(os.path.join(CLEAN_FOLDER, filename), "w", encoding="utf-8") as f:
            f.write(" ".join(filtered))

print("✅ Preprocessing completed. Clean articles ready.")
