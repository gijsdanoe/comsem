import utils



def main():
    sentence, token =  utils.read_files("Data/sentences.json","Data/tokens.json")
    print(sentence)
    


if __name__ == "__main__":
    main()

