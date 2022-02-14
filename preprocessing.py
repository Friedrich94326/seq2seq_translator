import re
import os
os.chdir("E:/IT_Iron_Man_Challenge/Machine Translation/seq2seq_translator")

# Load bilingual parallel corpus
data_path = "data/parallel_corpora/eng-span.txt"
with open(data_path, 'r', encoding = "utf-8") as f:
    lines = f.read().split('\n')

# text corpus
input_docs = list()
target_docs = list()
# all tokens including words, punctuations and <SOS>, <EOS>
input_tokens = set()
target_tokens = set()

for line in lines:
    # Input and target sentences are separated by tabs
    input_doc, target_doc = line.split('\t')
    # Appending each input sentence to input_docs
    input_docs.append(input_doc)
    # Splitting words from punctuation
    target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
    # Surrond each sentence by indicating symbols
    target_doc = "<SOS> " + target_doc + " <EOS>"

    target_docs.append(target_doc)

    # Now we split up each sentence into words
    # and add each unique word to our vocabulary set
    for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
        print(token)
        if token not in input_tokens:
            input_tokens.add(token)

    for token in target_doc.split():
        print(token)
        if token not in target_tokens:
            target_tokens.add(token)


input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)
print("num_encoder_tokens: {}".format(num_encoder_tokens)) # 18
print("num_decoder_tokens: {}".format(num_decoder_tokens)) # 27

try:
    max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
    max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])
except ValueError:
    pass