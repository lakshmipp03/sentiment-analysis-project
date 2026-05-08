import re
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Stopwords
stop_words = set(stopwords.words('english'))

# Clean text function
def clean_text(text):

    # Lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    filtered_words = []

    for word in words:

        if word not in stop_words:

            filtered_words.append(word)

    # Join words
    cleaned_text = " ".join(filtered_words)

    return cleaned_text