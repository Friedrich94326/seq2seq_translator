import numpy as np
import re
import os
from preprocessing import input_docs, target_docs, input_tokens, target_tokens, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length

# text vocabulary dictionaries for source and target languages
input_features_dict = dict([(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict([(token, i) for i, token in enumerate(target_tokens)])

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_features_dict = dict((i, token) for token, i in input_features_dict.items())
# Build out reverse_target_features_dict:
reverse_target_features_dict = dict((i, token) for token, i in target_features_dict.items())

# set up inputs and output for encoder and decoder
encoder_input_data = np.zeros(
    shape = (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
    dtype = "float32")
print("\nHere's the first item in the encoder input matrix:\n", encoder_input_data[0])

decoder_input_data = np.zeros(
    shape = (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype = "float32")
decoder_target_data = np.zeros(
    (len(target_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype = "float32")

# Examing shapes
print("shape of encoder input data: ", encoder_input_data.shape) # (11, 4, 18)
print("shape of decoder input data: ", decoder_input_data.shape) # (11, 12, 27)
print("shape of decoder target data: ", decoder_target_data.shape) # (11, 12, 27)


# One-hot encode input and output tokens
print("[INFO] One-hot encoding tokens...")
for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
    print("Encoder inputs:")
    for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
        print("timestep: {} - token: {}".format(timestep, token))
        encoder_input_data[line, timestep, input_features_dict[token]] = 1.

    print("\n\n[INFO] Decoder inputs and targets...")
    for timestep, token in enumerate(target_doc.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        print("(input) timestep: {} - token: {}".format(timestep, token))
        decoder_input_data[line, timestep, target_features_dict[token]] = 1.
        if timestep > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start token.
            print("(target) timestep:", timestep)
            # Assign 1. for the current line, previous timestep, & word
            # in decoder_target_data:
            decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.

    # save our data
    out_file_dir = "data"
    np.save(os.path.join(out_file_dir, "encoder_inputs.npy"), encoder_input_data)
    np.save(os.path.join(out_file_dir, "decoder_inputs.npy"), decoder_input_data)
    np.save(os.path.join(out_file_dir, "decoder_targets.npy"), decoder_target_data)


