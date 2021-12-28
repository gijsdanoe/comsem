import json

def read_data(file):
    """Read in data sets and returns sentences and labels"""
    docIds = []
    sentences = []
    token_dict = {}
    token_dict_temp = {}
    docIds_prev = ""
    with open(file, encoding='utf-8') as f:
        docId = ""
        for line in f:
            if "# newdoc id = " in line:
                docId = line.replace("# newdoc id = ","").replace("\n","")
                docIds.append(docId)
            elif "# raw sent = " in line:
                sentences.append(line.replace("# raw sent = ",""))
                if token_dict_temp:
                    token_dict[docIds_prev] = token_dict_temp
                token_dict_temp = {}
            elif "#" not in line and len(line) > 1:
                tokens = line.strip().split()
                token_dict_temp[tokens[0]] = [tokens[5]]
                docIds_prev = docId



    return docIds, sentences, token_dict
