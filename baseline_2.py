import utils
import nltk
import json
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
nltk.download('omw-1.4')


# def skip_token(token, tag):
#     if tag.startswith('NNP'):
#         return True
#     if token.isdigit():
#         return True
#     if tag.startswith('PRP'):
#         return True
#     return False


def wordnet_pos_code(tag):
    if tag.startswith('NN'):
        return wn.NOUN
    elif tag.startswith('VB'):
        return wn.VERB
    elif tag.startswith('JJ'):
        return wn.ADJ
    elif tag.startswith('RB'):
        return wn.ADV
    else:
        return None


def create_temp_lookupdict_fr_trainset(json_path):
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
    return {
        token: max(label_count.items(), key=lambda v: v[1])[0]
        for token, label_count in token_label_count.items()
    }


def baseline_2(sentence, lookup):
    new_tokens = {}

    tokens = word_tokenize(sentence)
    tokens_pos = nltk.pos_tag(tokens)

    for token, pos in tokens_pos:
        if utils.skip_token(token, pos):
            continue

        if token in lookup:
            new_tokens[token] = lookup[token]
        else:
            wn_pos = wordnet_pos_code(pos)
            if wn_pos:
                synsets = wn.synsets(token, pos=wn_pos)
                if synsets:
                    new_tokens[token] = synsets[0].name()
                else:
                    new_tokens[token] = "O"

            else:
                synsets = wn.synsets(token)
                if synsets:
                    new_tokens[token] = synsets[0].name()
                else:
                    new_tokens[token] = "O"

    return new_tokens


# def main():
#     token_label_count = create_temp_lookupdict_fr_trainset("Data/tokens_labels_train.json")
#     lookup = create_lookupdict_fr_trainset(token_label_count)
#     sentences_test_file = utils.read_file('Data/sentences_test.json')

#     for sid, sent in sentences_test_file.items():

#         print(baseline_2(sent, lookup))


# if __name__ == "__main__":
#     main()
