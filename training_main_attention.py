"""
Build an LSTM Seq2Seq model with attention mechanism
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, dot, concatenate, Activation
from tensorflow.keras.activations import tanh
from tensorflow.keras.models import Model
from preprocessing import num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length
from matplotlib import pyplot as plt
from pathlib import Path
import os
import time
# Using a GPU
# tf.debugging.set_log_device_placement(True)
from tensorflow.keras.utils import plot_model
from seq2seq_lstm import LuongAttention, Encoder, Decoder

# Load training data
encoder_input_data = np.load("data/encoder_inputs.npy")
decoder_input_data = np.load("data/decoder_inputs.npy")
decoder_target_data = np.load("data/decoder_targets.npy")

latent_dim = 256



# Encoder- calling Keras API
enc_layer_1 = LSTM(latent_dim, return_sequences = True, return_state = True, name = "1st_layer_enc_LSTM")
enc_layer_2 = LSTM(latent_dim, return_sequences = True, return_state = True, name = "2nd_layer_enc_LSTM")
enc_inputs = Input(shape = (max_encoder_seq_length, num_encoder_tokens), dtype = "float32")
enc_outputs_1, enc_h1, enc_c1 = enc_layer_1(enc_inputs)
enc_outputs_2, enc_h2, enc_c2 = enc_layer_2(enc_outputs_1)
enc_states = [enc_h1, enc_c1, enc_h2, enc_h2]

# Decoder- calling Keras API
dec_layer_1 = LSTM(latent_dim, return_sequences = True, return_state = True, name = "1st_layer_dec_LSTM")
dec_layer_2 = LSTM(latent_dim, return_sequences = True, return_state = False, name = "2nd_layer_dec_LSTM")
dec_dense = Dense(num_decoder_tokens, activation = "softmax")
dec_inputs = Input(shape = (max_decoder_seq_length, num_decoder_tokens), dtype = "float32")
dec_outputs_1, dec_h1, dec_c1 = dec_layer_1(dec_inputs, initial_state = [enc_h1, enc_c1])
dec_outputs_2 = dec_layer_2(dec_outputs_1, initial_state = [enc_h2, enc_c2])

### Attention Mechanism
# evaluate attention score
attention_scores = dot([dec_outputs_2, enc_outputs_2], axes = [2, 2])
attenton_weights = Activation("softmax")(attention_scores)
context_vec = dot([attenton_weights, enc_outputs_2], axes = [2, 1])
ht_context_vec = concatenate([context_vec, dec_outputs_2], name = "concatentated_vector")
attention_vec = Dense(latent_dim, use_bias = False, activation = "tanh", name = "attentional_vector")(ht_context_vec)
dec_outputs_final = Dense(num_decoder_tokens, use_bias = False, activation = "softmax")(attention_vec)

# Build a seq2seq model
my_seq2seq = Model([enc_inputs, dec_inputs], dec_outputs_final, name = "seq2seq_attention_Eng-Span_translator")
my_seq2seq.summary()
my_seq2seq.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2),
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)
plot_model(
    my_seq2seq,
    to_file = "output/eng-span_translator_v1_attention.png",
    dpi = 100,
    show_shapes = True,
    show_layer_names = True,
    # show_dtype = True
    )

batch_size = encoder_input_data.shape[0]
epochs = 500

# Start training
with tf.device("/GPU:0"):
    start = time.time()
    train_hist =  my_seq2seq.fit(
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
fig, axes = plt.subplots(1, 2, figsize = (13, 5))
fig.suptitle("Training History of My Seq2Seq Model")
plt.tight_layout()
axes[0].set_title("Loss")
axes[0].plot(train_hist.history["loss"], label = "train")
axes[0].plot(train_hist.history["val_loss"], label = "test")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("loss")
axes[0].legend(loc = "upper right")
axes[1].set_title("Accuracy")
axes[1].plot(train_hist.history["accuracy"], label = "train")
axes[1].plot(train_hist.history["val_accuracy"], label = "test")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("accuracy")
axes[1].legend(loc = "lower right")
plt.show()

# save model
out_model_dir = "models"
saveModel = input("Do you want to save the current model? (y/n)")
if saveModel == 'y':
    Path("models").mkdir(parents = True, exist_ok = True)
    my_seq2seq.save(os.path.join(out_model_dir, "eng-span_translator_v1_attention.h5"))
else:
    pass





