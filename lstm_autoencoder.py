import tensorflow as tf
import numpy as np
from data_preprocess import data_iterator

# english
# onehot_tok_idx = np.load('data/en_onehot')
# en_file_path = "data/english_subtitles.gz"


# french
onehot_tok_idx = np.load('data/fr_onehot.npy').item()
fr_file_path = "data/french_subtitles.gz"

# Build LSTM graph


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

vocab_size = len(onehot_tok_idx)
num_layers = 4
num_steps = 20
batch_size = 1
hidden_size = 500

# batch_size x num_steps x vocab_size with post-padding
raw_sequence = tf.placeholder(tf.float16, shape=(
    batch_size, num_steps, vocab_size), name="placeholder_raw_sequence")

# encoder_inputs must be a 3D tensor [num_steps x batch_size x vocab_size]
encoder_inputs = tf.transpose(raw_sequence, [1, 0, 2])

# decoder_inputs must be a list of 2D tensors [batch_size x
# vocab_size] of length num_steps
decoder_inputs = tf.unstack(encoder_inputs, axis=0)
decoder_inputs = (
    [tf.zeros_like(decoder_inputs[0], name="GO")] + decoder_inputs[:-1])

# labels must be a list of 1D tensors [batch_size] of length num_steps
raw_labels = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name="placeholder_raw_labels")
labels = tf.unstack(raw_labels, axis=1)

# tf.reduce_max() collapses one-hot dimension to max (1 if used step or 0 if unused step)
# tf.sign() to convert max value to 1
# tensor of shape: [ batch_size x num_steps ]
used_frames = tf.sign(tf.reduce_max(tf.abs(raw_sequence), axis=2))

# tf.reduce_sum() collapses indicator dimension (1 if used step or 0 if unused step) by summing values
# tf.cast() to convert to type tf.int32
# tensor of shape: [ batch_size ]
encoder_sequence_lengths = tf.cast(tf.reduce_sum(
    used_frames, reduction_indices=1), tf.int32)

# tf.transpose() takes transpose
# tf.cast() to convert to type tf.float16
# tf.unstack() to create list of 1D tensors
# list of 1D tensors of shape: [ batch_size ] of length num_steps
decoder_weights = tf.unstack(tf.cast(used_frames, tf.float16), axis=1)

# labels = encoder_inputs

lstm = tf.contrib.rnn.BasicLSTMCell(
    hidden_size, forget_bias=0.0, state_is_tuple=True)
stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    [lstm] * num_layers, state_is_tuple=True)

initial_state = stacked_lstm.zero_state(batch_size, dtype=tf.float16)

encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    stacked_lstm, raw_sequence, initial_state=initial_state, dtype=tf.float16, sequence_length=encoder_sequence_lengths)

outputs, state = tf.contrib.legacy_seq2seq.rnn_decoder(
    decoder_inputs=decoder_inputs, initial_state=encoder_state, cell=stacked_lstm)

loss = tf.contrib.legacy_seq2seq.sequence_loss(logits=outputs, targets=labels, weights=decoder_weights)

optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss)

print("graph loaded")

iter_ = data_iterator(fr_file_path, onehot_tok_idx, 1,batch_size, num_steps)

print("iterator loaded")

# saver = tf.train.Saver()

init = tf.global_variables_initializer()

print("variables initialized")

with tf.Session() as sess:
    sess.run(init)
    for i in range(200000):
        sequences_batch,labels_batch = iter_.__next__()

        if (i + 1) % 100 == 0:
            train_accuracy = loss.eval(session=sess, feed_dict={
                                       raw_sequence: sequences_batch, raw_labels: labels_batch})
            print("step %d, training loss %g" % (i + 1, train_accuracy))

        optimizer.run(session=sess, feed_dict={raw_sequence: sequences_batch, raw_labels: labels_batch})
