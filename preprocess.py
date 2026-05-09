import re
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# English stopwords
stop_words = set(stopwords.words('english'))

# Clean text function
def clean_text(text):

    # Lowercase
    text = text.lower()

    # REMOVE ONLY SPECIAL SYMBOLS
    # KEEP multilingual letters
    text = re.sub(r'[^\w\s]', '', text)

    try:

        # Tokenization
        words = word_tokenize(text)

    except:

        return text

    filtered_words = []

    for word in words:

        if word not in stop_words:

            filtered_words.append(word)

    cleaned_text = " ".join(filtered_words)

    return cleaned_text