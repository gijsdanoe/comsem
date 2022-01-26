import json
import pandas as pd
import string

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
    if token.lower().startswith('whose'):
        return True
    if token.startswith('Consuming~Kids'):
        return True
    return False


def write_to_csv(c1, c2, filename):
    df = pd.DataFrame()
    df['Token'] = c1
    df['Predict'] = c2
    df.to_csv(OUTPUT_DIR + filename + ".csv", index=False)
