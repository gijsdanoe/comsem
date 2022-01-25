import utils
import nltk
import json
from nltk.corpus import wordnet as wn
nltk.download('omw-1.4')


def get_tokens_labels(file):
    ''' get tokens and their labels from data file and
    save them to a json file'''

    Ids, sentence, doc_dict = utils.read_data(file)
    utils.write2Json(doc_dict, 'Data/tokens_labels.json')


def create_temp_lookupdict_fr_trainset(json_file):
    lookup_dict = {}
    tokens_json = utils.read_file("Data/tokens.json")
    print(tokens_json)


    return lookup_dict


def create_lookupdict_fr_trainset(dict)
    pass

def save_to_csv(dict):
    pass


def main():





if __name__ == "__main__":
    main()
