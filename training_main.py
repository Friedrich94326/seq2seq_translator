"""
Build an LSTM Seq2Seq model without attention mechanism
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
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
encoder_inputs = Input(shape = (max_encoder_seq_length,))
embedded_enc_inputs = Embedding(num_encoder_tokens, latent_dim, name = "embedding_layer_encoder")(encoder_inputs)
encoder_outputs, state_hidden, state_cell = LSTM(latent_dim, return_state = True, name = "LSTM_encoder")(embedded_enc_inputs)
encoder_states = [state_hidden, state_cell]
print("shape of state_hidden: ", state_hidden.shape) # (None, 256)
print("shape of state_cell: ", state_cell.shape) # (None, 256)
print("encoder states:", encoder_states)
os.system("pause") # [<tf.Tensor 'lstm/PartitionedCall:2' shape=(None, 256) dtype=float32>, <tf.Tensor 'lstm/PartitionedCall:3' shape=(None, 256) dtype=float32>]

# Decoder- calling Keras API
decoder_inputs = Input(shape = (max_decoder_seq_length, ))
embedded_dec_inputs = Embedding(num_decoder_tokens, latent_dim, name = "embedding_layer_decoder")(decoder_inputs)
decoder_outputs, decoder_state_hidden, decoder_state_cell = LSTM(latent_dim, return_sequences = True, return_state = True, name = "LSTM_decoder")(embedded_dec_inputs, initial_state = encoder_states)
decoder_outputs = Dense(num_decoder_tokens, activation = "softmax")(decoder_outputs)
print("shape of decoder_state_hidden: ", decoder_state_hidden.shape) # (None, 256)
print("shape of decoder_state_cell: ", decoder_state_cell.shape) # (None, 256)
os.system("pause")

# Build a seq2seq model
my_seq2seq = Model([encoder_inputs, decoder_inputs], decoder_outputs, name = "seq2seq_Eng-Span_translator")
my_seq2seq.summary()
my_seq2seq.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)
plot_model(
    my_seq2seq,
    to_file = "output/eng-span_translator_v1.png",
    dpi = 100,
    show_shapes = True,
    show_layer_names = True,
    # show_dtype = True,
    )

batch_size = 11
epochs = 100

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
                        validation_split = 0.2,
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
    my_seq2seq.save(os.path.join(out_model_dir, "eng-span_translator_v1.h5"))
else:
    pass





