import utils
import nltk
import json
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')



def create_temp_lookupdict_fr_trainset(json_path):
    '''Collect all the labels of a token, and the
       count of its labels. Return a dictionary
       with key is token, value is a dictionary
       with key is the label, and value is its count. '''

    token_label_count = {}
    sid_tokens_labels = utils.read_file(json_path)

    for tokens_labels in sid_tokens_labels.values():
        tagged_tokens = nltk.pos_tag(tokens_labels.keys())

        for token, pos in tagged_tokens:
            label = tokens_labels[token][0]

            if utils.skip_token(token, pos):
                continue

            if token not in token_label_count:
                token_label_count[token] = {}
            if label not in token_label_count[token]:
                token_label_count[token][label] = 0

            token_label_count[token][label] += 1

    return token_label_count


def create_lookupdict_fr_trainset(token_label_count):
    '''Take the max count of the label of a token.
       Return a dictionary with key is token, and
       value is its most frequent label.'''

    return {
        token: max(label_count.items(), key=lambda v: v[1])[0]
        for token, label_count in token_label_count.items()
    }


def baseline_2(sentence, lookup):
    '''Filter skipped token, use the lookup
       dictionary to label tokens in the input
       sentence. Return a dictionary with key
       is token, value is its label.'''

    result_dict = {}

    tokens = word_tokenize(sentence)
    tokens_pos = nltk.pos_tag(tokens)

    for token, pos in tokens_pos:
        if utils.skip_token(token, pos):
            continue

        if token in lookup:
            result_dict[token] = lookup[token]
        else:
            wn_pos = utils.wordnet_pos_code(pos)
            if wn_pos:
                synsets = wn.synsets(token, pos=wn_pos)
                if synsets:
                    result_dict[token] = synsets[0].name()
                else:
                    result_dict[token] = "O"

            else:
                synsets = wn.synsets(token)
                if synsets:
                    result_dict[token] = synsets[0].name()
                else:
                    result_dict[token] = "O"

    return result_dict


