from bs4 import BeautifulSoup
import contractions
import unidecode
import requests
import string
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# ## Import Training Data (Text Corpus) via input URL:
url = 'https://www.gutenberg.org/files/57354/57354-h/57354-h.htm'
headers = {
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/W.X.Y.Z Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
          }
req = requests.get(url, headers)
soup = BeautifulSoup(req.content, 'html.parser')
training_text = soup.get_text()
# print(training_text)

## Import Training Data (Text Corpus) via input File:
# with open('training-text.txt', encoding='utf-8') as input_file:
#     training_text = input_file.read()
# sample_training_text = training_text[0:2000]
# print(sample_training_text)

## Clean Training Data: (Text Pre-processing)
def text_preprocessor(text):
    ###########################################
    soup = BeautifulSoup(text, 'html.parser') ## Remove HTML
    text = soup.get_text(separator=' ') ## Remove HTML
    ###########################################
    text = text.lower() ## Lowercase Characters
    text = contractions.fix(text) ## Expand Contractions ("don't" -> "do not")
    text = text.translate(str.maketrans('', '', string.punctuation)) ## Remove Punctuation Characters
    text = re.sub(r'(’)','', text) # Remove known characters
    text = re.sub(r'[0-9]+', '', text) ## Remove Numerical Characters
    text = unidecode.unidecode(text) ## Normalized accented characters (ñ -> n)
    ###########################################
    text = word_tokenize(text) ## Tokenize Text    
    # stop_words = set(stopwords.words('english')) ## Get Stop Words
    # stop_words_exclusion = ['no','not','nor'] ## Stop Word Exclusion List
    # stop_words = [word for word in stop_words if word not in stop_words_exclusion] ## Remove Stop Word Exclusions from Stop Words
    # text = [word for word in text if word not in stop_words] ## Remove Stop Words
    ###########################################
    # ps = PorterStemmer() ## Stemming: ['wait', 'waiting', 'waited', 'waits'] -> 'wait'
    # text = [ps.stem(word) for word in text] ## Apply Word Stemming
    wnl = WordNetLemmatizer() ## Lemmatization: 'studies' -> 'study'; 'studying' -> 'studying'
    text = [wnl.lemmatize(word) for word in text] ## Apply Word Lemmatization
    return text

training_text = text_preprocessor(training_text)
# print(training_text)
