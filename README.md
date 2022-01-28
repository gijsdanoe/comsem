# Computational Sementantics
## To test our system just run WordSenseDisambiguation.py (also check argparse)
### To use your own file, please change it to the same format as the example file 'test_sentences.txt'
### For testing the other scripts you need to gave 'en_core_web_sm'.

# Evaluation
### To perform evaluation you first need to run create_files.py to read the raw dataset (train.txt, test.txt, dev.txt or eval.txt )
### After runing this script you will get 2 json files: 1) tokens_test.json 2) sentence_test.json (if test.txt was chosen as dataset)
### Next run sim_checkers.py and give that dataset that you used as argument for the argparser (-d). This script will produce an output.json containing the wordsense labels produced by our system.
### To check accuracy run test_system.py and give that dataset that you used as argument for the argparser (-d).
