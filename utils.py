import json
import pandas as pd
import string
import nltk

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
nltk.download('omw-1.4')


OUTPUT_DIR = 'Output/'

def read_data(file):
    """Read in data sets and returns sentences and labels"""
    docIds = []
    sentences = []
    token_dict = {}
    token_dict_temp = {}
    docIds_prev = ""
    sentence = ""
    add_sent = False
    with open(file, encoding='utf-8') as f:
        docId = ""
        for line in f:
            if "# newdoc id = " in line:
                docId = line.replace("# newdoc id = ","").replace("\n","")
                docIds.append(docId)
                add_sent = True
                if token_dict_temp:
                    token_dict[docIds_prev] = token_dict_temp
                token_dict_temp = {}
                if len(sentence) > 1:
                    sentences.append(sentence)
                    sentence = ""

           # elif "# raw sent = " in line:
            #    if add_sent:
            #        sentences.append(line.replace("# raw sent = ","").replace("\n",""))
             #       add_sent = False

    #            if token_dict_temp:
      #              token_dict[docIds_prev] = token_dict_temp
      #          token_dict_temp = {}

            elif "#" not in line and len(line) > 1:
                tokens = line.strip().split()
                if len(sentence) <1:
                    sentence+= tokens[0]
                else:
                    sentence+= " "+ tokens[0]
                token_dict_temp[tokens[0]] = [tokens[5]]
                docIds_prev = docId
        token_dict[docIds_prev] = token_dict_temp

        token_dict[docIds_prev] = token_dict_temp
        sentences.append(sentence)

    return docIds, sentences, token_dict


def read_files(token_file, sentence_file):
    token_dict = json.load(open(token_file))
    sentence_dict = json.load(open(sentence_file))
    return token_dict, sentence_dict
    

def read_file(json_file):
    file_js = json.load(open(json_file))
    return file_js


def write2Json(doc_dict, path):
    with open(path, 'w') as fp:
        json.dump(doc_dict, fp)


def skip_token(token, tag):
    '''Return a boolean value if the input token
       meets these condition or not.'''
    if tag.startswith('NNP'):
        return True
    if tag.startswith('PRP'):
        return True
    if tag.startswith('DT'):
        return True
    if tag.startswith('WP'):
        return True

    if token[0].isdigit():
        return True
    if token.startswith('http'):
        return True
    if token.startswith('//'):
        return True
    if token.startswith('`'):
        return True
    if token in string.punctuation :
        return True

    # filter some words which are not consistent
    # in pos tags of the tokens from sentences,
    # and the tokens from the token column in the data.
    if token.lower() == 'whose':
        return True
    if token =='Consuming~Kids':
        return True
    if token == 'Mary':
        return True
    if token == 'either':
        return True
    if token == 'that':
        return True
    if token == 'Taninna':
        return True
    if token == 'Chamber~of~Deputies':
        return True

    return False


def write_to_csv(c1, c2, filename):
    '''Write 2 lists to a csv file without headers for columns'''
    df = pd.DataFrame()
    df['Token'] = c1
    df['Predict'] = c2
    df.to_csv(OUTPUT_DIR + filename + ".csv", index=False, header=False)


def wordnet_pos_code(tag):
    '''Convert pos tag to wordnet pos code'''
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

ef write2Json(doc_dict, path):
    with open(path, 'w') as fp:
        json.dump(doc_dict, fp)

def read_data2(file):
    """Read baseline data"""
    base = []
    gold = []
    token = []
    labels = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split("\t")
            token.append(tokens[0])
            gold.append(tokens[1])
            base.append(tokens[2])
            labels.append(tokens[3])
    return token, gold, base, labels


def read_data_from_user(file):
    """User data"""
    ids = []
    sentences = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split("\t")
            ids.append(tokens[0])
            sentences.append(tokens[1])
    return ids, sentences
