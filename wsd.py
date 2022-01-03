import utils
import nltk
import json
import clauses

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('omw-1.4')


OUTPUT_DIR = "Output/"
DATA_DIR = "Data/"


def get_sentences(doc_dict):
    '''Get value from a dictionary, return list of sentences'''
    sentences = list(doc_dict.values())
    return sentences


def get_tokens(sentence):
    '''Tokenize input sentence, return a list of tokens.'''
    tokens = word_tokenize(sentence)

    return tokens


def main():
    file = DATA_DIR + "sentences.json"
    doc_dict = utils.read_file(file)
    sentences = get_sentences(doc_dict)
    tokens = [get_tokens(sent) for sent in sentences]
    predicted_labels = []
    for sent in sentences:
        for token in get_tokens(sent):
            if len(list(wn.synsets(token))) == 0:
                predicted_labels.append("O")

            else:
                clauses.get_wn_definition(token)





if __name__ == '__main__':
    main()

