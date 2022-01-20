import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet


def get_wn_definition(token):
    sentence_dict = {}
    sentences = []
    result = wordnet.synsets(token)
    if not result:
        print('No found: ', token)

    for item in result:
        sentence_dict[item.name()] = item.definition()
    return sentence_dict


def main():
    print(get_clauses("arm"))


if __name__ == "__main__":
    main()



