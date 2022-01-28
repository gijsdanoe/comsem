"""
This script accepts a .txt file with sentences and perform wordsense disambiguation
"""

import nltk
import spacy
import utils
import argparse
import sim_checkers
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("en_core_web_sm")


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", default='test_sentences.txt', type=str, help="Input file with sentences")
    parser.add_argument("-g", "--gloss", action="store_true", help="Use gloss only rules")
    args = parser.parse_args()
    
    return args


def main():
    args = create_arg_parser()
    ids, sentences = utils.read_data_from_user(f"Data/{args.input_file}")
    sim_checkers.get_wordsense(ids, sentences,args.gloss)


if __name__ == "__main__":
    main()
