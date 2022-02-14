#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   seq2seq_lstm.py
@Date    :   10.03.2021
@Author  :   Friedrich Cheng
@Version :   1.0
@Contact :   codingfriedrich94326@gmail.com
@Last Modified by:   Friedrich
@Last Modified time: 10.03.2021
@Description:
    Here we define classes which respectively represent layers Encoder_LSTM, Decoder_LSTM, Attention.
    Also, we define classes represent model TrainTranslator (in training phase), Translator (in inference phase).
'''

# here put the import lib
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, LSTM, Embedding, Dot, Activation, concatenate
import pydot as pyd
from tensorflow.keras.utils import plot_model
import os
import time


class LuongAttention(Layer):
    """
    Luong attention layer.
    """
    def __init__(self, latent_dim, tgt_wordEmbed_dim):
        super().__init__()
        self.AttentionFunction = Dot(axes = [2, 2], name = "attention_function")
        self.SoftMax = Activation("softmax", name = "softmax_attention")
        self.WeightedSum = Dot(axes = [2, 1], name = "weighted_sum")
        self.dense_tanh = Dense(latent_dim, use_bias = False, activation = "tanh", name = "dense_tanh")
        self.dense_softmax = Dense(tgt_wordEmbed_dim, use_bias = False, activation = "softmax", name = "dense_softmax")

    def call(self, inputs):
        # unpack inputs
        enc_outputs_top, dec_outputs_top = inputs
        print("LuongAttention]\n shapes of enc_outputs_top: {}, dec_outputs_top: {}".format(enc_outputs_top.shape, dec_outputs_top.shape))
        # os.system("pause")
        attention_scores = self.AttentionFunction([dec_outputs_top, enc_outputs_top])
        attenton_weights = self.SoftMax(attention_scores)
        print("attention weights - shape: {}".format(attenton_weights.shape))
        context_vec = self.WeightedSum([attenton_weights, enc_outputs_top])
        print("context vector - shape: {}".format(context_vec.shape))
        ht_context_vec = concatenate([context_vec, dec_outputs_top], name = "concatentated_vector")
        print("ht_context_vec - shape: {}".format(ht_context_vec.shape))
        attention_vec = self.dense_tanh(ht_context_vec)
        print("attention_vec - shape: {}".format(attention_vec.shape))
        return attention_vec



class Encoder(Layer):
    """
    2-layer Encoder LSTM with/ without attention mechanism.
    """
    def __init__(self, latent_dim, src_wordEmbed_dim, src_max_seq_length, withAttention = False):
        super().__init__()
        # self.inputs = Input(shape = (src_max_seq_length, src_wordEmbed_dim), name = "encoder_inputs")
        self.latent_dim = latent_dim
        self.embedding_dim = src_wordEmbed_dim
        self.max_seq_length = src_max_seq_length
        self.lstm_input = LSTM(units = latent_dim, return_sequences = True, return_state = True, name = "1st_layer_enc_LSTM")
        self.lstm = LSTM(units = latent_dim, return_sequences = False, return_state = True, name = "2nd_layer_enc_LSTM")
        self.lstm_return_seqs = LSTM(units = latent_dim, return_sequences = True, return_state = True, name = "2nd_layer_enc_LSTM")
        self.withAttention = withAttention

    def call(self, inputs):
        print("[Encoder]\n inputs shape: {}".format(inputs.shape))
        # os.system("pause")
        outputs_1, h1, c1 = self.lstm_input(inputs)
        if self.withAttention:
            outputs_2, h2, c2 = self.lstm_return_seqs(outputs_1)
        else:
            outputs_2, h2, c2 = self.lstm(outputs_1)
        states = [h1, c1, h2, h2]
        return outputs_2, states



class Decoder(Layer):
    """
    2-layer Decoder LSTM with/ without attention mechanism.
    """
    def __init__(self, latent_dim, tgt_wordEmbed_dim, tgt_max_seq_length, withAttention = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = tgt_wordEmbed_dim
        self.max_seq_length = tgt_max_seq_length
        self.lstm_input = LSTM(units = latent_dim, return_sequences = True, return_state = True, name = "1st_layer_dec_LSTM")
        self.lstm_return_no_states = LSTM(units = latent_dim, return_sequences = True, return_state = False, name = "2nd_layer_dec_LSTM")
        self.lstm = LSTM(units = latent_dim, return_sequences = True, return_state = True, name = "2nd_layer_dec_LSTM")
        self.dense = Dense(tgt_wordEmbed_dim, activation = "softmax", name = "softmax_dec_LSTM")
        self.withAttention = withAttention

    def call(self, inputs):
        # unpack inputs
        dec_inputs, enc_outputs_top, enc_states = inputs
        print("[Decoder]\n dec_inputs shape: {} - enc_outputs_top shape: {}".format(dec_inputs.shape, enc_inputs.shape))

        # unpack encoder states [h1, c1, h2, c2]
        enc_h1, enc_c1, enc_h2, enc_c2 = enc_states
        print("Shapes of enc_h1: {}, enc_c1: {}, enc_h2: {}, enc_c2: {}".format(enc_h1.shape, enc_c1.shape, enc_h2.shape, enc_c2.shape))
        # os.system("pause")
        outputs_1, h1, c1 = self.lstm_input(dec_inputs, initial_state = [enc_h1, enc_c1])
        if self.withAttention:
            # instantiate Luong attention layer
            attention_layer = LuongAttention(latent_dim = self.latent_dim, tgt_wordEmbed_dim = self.max_seq_length)

            dec_outputs_top = self.lstm_return_no_states(outputs_1, initial_state = [enc_h2, enc_c2])
            attention_vec = attention_layer((enc_outputs_top, dec_outputs_top))
            outputs_final = self.dense(attention_vec)
        else:
            outputs_2, h2, c2 = self.lstm(outputs_1, initial_state = [enc_h2, enc_c2])
            outputs_final = self.dense(outputs_2)
        print("outputs_final - shape: {}".format(outputs_final.shape))
        return outputs_final



class My_Seq2Seq(Model):
    """
    2-Layer LSTM Encoder-Decoder with/ without Luong attention mechanism.
    """
    def __init__(self, latent_dim, src_wordEmbed_dim, src_max_seq_length, tgt_wordEmbed_dim, tgt_max_seq_length, model_name = None, withAttention = False,
 input_text_processor = None, output_text_processor = None):
        super().__init__(name = model_name)
        self.encoder = Encoder(latent_dim, src_wordEmbed_dim, src_max_seq_length, withAttention = withAttention)
        self.decoder = Decoder(latent_dim, tgt_wordEmbed_dim, tgt_max_seq_length, withAttention = withAttention)
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.withAttention = withAttention


    def call(self, inputs):
        # unpack inputs
        enc_inputs, dec_inputs = inputs
        print("[MySeq2Seq]\n enc_inputs shape: {} - dec_inputs shape: {}".format(enc_inputs.shape, dec_inputs.shape))
        # os.system("pause")
        enc_outputs, enc_states = self.encoder(enc_inputs)
        dec_outputs = self.decoder(inputs = (dec_inputs, enc_outputs, enc_states))
        return dec_outputs

    def plot_model_arch(self, enc_inputs, dec_inputs, outfile_path = None):
        tmp_model = Model(inputs = [enc_inputs, dec_inputs], outputs = self.call((enc_inputs, dec_inputs)))
        plot_model(tmp_model, to_file = outfile_path, dpi = 100, show_shapes = True, show_layer_names = True)


if __name__ == "__main__":
    # show curretn working directory
    os.chdir("E:/IT_Iron_Man_Challenge/Machine Translation/seq2seq_translator")
    print("[INFO] cwd: ", os.getcwd())

    # hyperparameters
    src_wordEmbed_dim = 18
    src_max_seq_length = 4
    tgt_wordEmbed_dim = 27
    tgt_max_seq_length = 12
    latent_dim = 256

    # preparing data
    enc_inputs = Input(shape = (src_max_seq_length, src_wordEmbed_dim))
    dec_inputs = Input(shape = (tgt_max_seq_length, tgt_wordEmbed_dim))


    seq2seq = My_Seq2Seq(latent_dim, src_wordEmbed_dim, src_max_seq_length, tgt_wordEmbed_dim, tgt_max_seq_length, withAttention = True, model_name = "seq2seq_no_attention")
    dec_outputs = seq2seq(
        [Input(shape = (src_max_seq_length, src_wordEmbed_dim)), Input(shape = (tgt_max_seq_length, tgt_wordEmbed_dim))]
        )
    print("model name: {}".format(seq2seq.name))
    seq2seq.summary()
    seq2seq.plot_model_arch(enc_inputs, dec_inputs, outfile_path = "output/seq2seq_LSTM_with_attention.png")
    seq2seq.compile(
        optimizer = "adam",
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )

    # Load training data
    encoder_input_data = np.load("data/encoder_inputs.npy")
    decoder_input_data = np.load("data/decoder_inputs.npy") # shape:
    decoder_target_data = np.load("data/decoder_targets.npy") # shape: (11, 12, 27)
    print("decoder_target_data shape: {}".format(decoder_target_data.shape))
    os.system("pause")

    batch_size = 50
    epochs = 100

    # tf.random.set_seed(5)
    # encoder_input_data_rand = tf.random.normal([11, src_max_seq_length, src_wordEmbed_dim], 0, 1, tf.float32)
    # decoder_input_data_rand = tf.random.normal([11, tgt_max_seq_length, tgt_wordEmbed_dim], 0, 1, tf.float32)
    # decoder_target_data_rand = tf.random.normal([11, 12, 27], 0, 1, tf.float32)
    withAttention = True
    encoder = Encoder(latent_dim, src_wordEmbed_dim, src_max_seq_length, withAttention = withAttention)
    decoder = Decoder(latent_dim, tgt_wordEmbed_dim, tgt_max_seq_length, withAttention = withAttention)

    enc_outputs, enc_states = encoder(enc_inputs)
    dec_outputs = decoder((dec_inputs, enc_outputs, enc_states))
    my_seq2seq_v2 = Model(inputs = [enc_inputs, dec_inputs], outputs = dec_outputs, name = "seq2seq_with_attention")
    my_seq2seq_v2.compile(
            optimizer = "adam",
            loss = "categorical_crossentropy",
            metrics = ["accuracy"]
        )

    with tf.device("/GPU:0"):
        start = time.time()
        train_hist =  my_seq2seq_v2.fit(
                            x = [encoder_input_data, decoder_input_data],
                            y = decoder_target_data,
                            batch_size = batch_size,
                            epochs = epochs,
                            shuffle = True,
                            verbose = 1,
                            validation_split = 0.2
                            )
        print("Training done. Time spent: {:.2f} s with a GPU".format(time.time() - start))

    # Review training history
    print("All history keys: {}".format(train_hist.history.keys()))
