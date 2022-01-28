"""
This script will search wordnet for the glosses and examples sentences of a token
"""
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet
import spacy
sp = spacy.load('en_core_web_sm')


def get_wn_definition(token, POS):
    """
    This function will search wordnet for gloss of the token given as argument filtered by the POS given as argument
    
    This function returns a dict containing the sense as key and its gloss as value
    """
    sentence_dict = {}
    sentences = []
    token = token.replace("~","_") #handle phrases
    
    POS_formatted = getPOSTags(POS)
    result = wordnet.synsets(token, pos =getPOSTags(POS) )
    
    for item in result:
        sentence_dict[item.name()] = item.definition()
    return sentence_dict


def get_wn_definition_examples(token, POS):
    """
    This function will search wordnet for gloss or example sentences of the token given as argument filtered by the POS given as argument
    
    This function returns a dict containing the sense as key and its gloss or example sentence as value
    """
    sentence_dict = {}
    sentences = []
    token = token.replace("~","_")
    POS_formatted = getPOSTags(POS)

    result = wordnet.synsets(token, pos =getPOSTags(POS) )
    for item in result:
        examples = item.examples()
        if len(examples) < 1:
            sentence_dict[item.name()] = item.definition()
        else:
            version = 1# this is just to make key unique (because there can be multiple example sentences for 1 sense)
            for x in examples:
                sentence_dict[item.name()+"_"+f"{version}"] = x
                version += 1
            
    return sentence_dict



def getPOSTags(POS):
    """
    This function will get correct POS format for wordnet
    """
    if POS == "NOUN":
        return wordnet.NOUN
    elif POS == "VERB":
        return wordnet.VERB
    elif POS == "ADV":
        return wordnet.ADV
    elif POS == "ADJ":
        return wordnet.ADJ
    else:
        return "No"


def main():
    #Example
    print(get_wn_definition("roared","ADV"))


if __name__ == "__main__":
    main()
