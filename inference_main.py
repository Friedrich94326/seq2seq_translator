import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.python.eager.context import device
from training_main import encoder_inputs, decoder_inputs, encoder_states, decoder_lstm, decoder_dense
from preprocessing import max_decoder_seq_length, num_decoder_tokens, input_docs, target_docs
from prepare_training_data import reverse_target_features_dict, target_features_dict, encoder_input_data

training_model = load_model("models/eng-span_translator_v1.h5")
encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]

# Building the encoder test model:
encoder_model = Model(encoder_inputs, encoder_states)

latent_dim = 256
# Building the two decoder state input layers:
decoder_state_input_hidden = Input(shape = (latent_dim, ))

decoder_state_input_cell = Input(shape = (latent_dim, ))

# Put the state input layers into a list:
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

# Call the decoder LSTM:
decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state = decoder_states_inputs)
decoder_states = [state_hidden, state_cell]

# Redefine the decoder outputs:
decoder_outputs = decoder_dense(decoder_outputs)

# Build the decoder test model:
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

def translate_sentence(test_input):
    # Encode the input as state vectors:
    encoder_states_value = encoder_model.predict(test_input)
    # Set decoder states equal to encoder final states
    decoder_states_value = encoder_states_value
    # Generate empty target sequence of length 1:
    # (batch size, number of tokens to start with, number of tokens in our target vocabulary)
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # Populate the first token of target sequence with the start token:
    target_seq[0, 0, target_features_dict["<SOS>"]] = 1.

    decoded_sentence = ''
    stop_condition = False
    while not stop_condition:
        # Run the decoder model to get possible
        # output tokens (with probabilities) & states
        output_tokens, new_decoder_hidden_state, new_decoder_cell_state = decoder_model.predict(
        [target_seq] + decoder_states_value)

    # Choose token with highest probability
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_token = reverse_target_features_dict[sampled_token_index]
    decoded_sentence += ' ' + sampled_token

    # Exit condition: either hit max length or find <EOS>
    if (sampled_token == "<EOS>" or len(decoded_sentence) > max_decoder_seq_length):
        stop_condition = True
    # Update the target sequence (of length 1)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, sampled_token_index] = 1.

    # Update states
    decoder_states_value = [new_decoder_hidden_state, new_decoder_cell_state]
    return decoded_sentence



def translate_sentence_1(input_seq):
    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first token of target sequence with <SOS>
    target_seq[0, 0, target_features_dict["<SOS>"]] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        decoded_sentence += sampled_token

        # Exit condition: either hit max length or find <EOS>
        if (sampled_token == "<EOS>" or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

with device("/GPU:0"):
    for seq_idx in range(10):
        test_input = encoder_input_data[seq_idx: seq_idx + 1]
        translated_sentence = translate_sentence(test_input)
        print("---------------------------")
        print("Source sentence:", input_docs[seq_idx])
        print("Translated sentence:", translated_sentence)
        print("Ground truth target sentence: ", target_docs[seq_idx])