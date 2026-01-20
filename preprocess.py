import re
import contractions  # pip install contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Initialize
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Map NLTK POS tags to WordNet tags for accurate lemmatization
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Advanced preprocessing function
def preprocess_text(text, min_len=2):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Expand contractions ("don't" -> "do not")
    text = contractions.fix(text)
    
    # 3. Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    
    # 4. Keep letters and numbers only
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    
    # 5. Tokenize
    tokens = word_tokenize(text)
    
    # 6. POS tagging
    pos_tags = pos_tag(tokens)
    
    # 7. Lemmatize, remove stopwords and short tokens
    processed_tokens = [
        lemmatizer.lemmatize(t, get_wordnet_pos(pos))
        for t, pos in pos_tags
        if t not in stop_words and len(t) > min_len
    ]
    
    return processed_tokens


