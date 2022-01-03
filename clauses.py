from nltk.corpus import wordnet

def get_clauses(token):
    sentence_dict = {}
    sentences = []
    result = wordnet.synsets(token)
    if not result:
        print('No found: ', word)

    for item in result:
        sentence_dict[item.name()] = item.examples()
    return sentence_dict




def main():
    print("")


if __name__ == "__main__":
    main()


