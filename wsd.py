import utils
import nltk
import json

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('omw-1.4')


def get_synsets(token):
    if len(list(wn.synsets(token))) > 1
