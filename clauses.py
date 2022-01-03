from nltk.corpus import wordnet


def get_wn_definition(token):
    sentence_dict = {}
    sentences = []
    result = wordnet.synsets(token)
    if not result:
        print('No found: ', word)

    for item in result:
        sentence_dict[item.name()] = item.definition()
    return sentence_dict


# def main():
#     print(get_wn_definition("dog"))


# if __name__ == "__main__":
#     main()


