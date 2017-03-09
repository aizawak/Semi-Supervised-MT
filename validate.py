import tensorflow as tf
import numpy as np
from data_preprocess import data_iterator

#### LOAD GRAPH

# Build LSTM graph
def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

vocab_size = len(onehot_tok_idx)
num_layers = 4
num_steps = 48
batch_size = 384
hidden_size = 250

# tensor of shape [ batch_size x num_steps x vocab_size ] with post-padding
encoder_inputs = tf.placeholder(tf.float32, shape=(
    batch_size, num_steps, vocab_size), name="placeholder_encoder_inputs")

# tensor of shape [ batch_size x num_steps ]
mask = tf.sign(tf.reduce_max(tf.abs(encoder_inputs), axis=2))

# tensor of shape [ batch_size ]
encoder_lengths = tf.cast(tf.reduce_sum(
    mask, reduction_indices=1), tf.int32)

# list of 2D tensors [ batch_size x vocab_size ] of length num_steps
decoder_inputs = tf.unstack(tf.transpose(encoder_inputs, [1, 0, 2]), axis=0)
decoder_inputs = (
    [tf.zeros_like(decoder_inputs[0], name="GO")] + decoder_inputs[:-1])

decoder_weights = tf.Variable(tf.truncated_normal(
    [hidden_size, vocab_size], stddev=0.05, dtype=tf.float32))
decoder_bias = tf.Variable(
    tf.constant(1, shape=[vocab_size], dtype=tf.float32))

# list of 1D tensors [ batch_size ] of length num_steps
raw_labels = tf.placeholder(tf.int32, shape=(
    batch_size, num_steps), name="placeholder_raw_labels")
targets = tf.unstack(raw_labels, axis=1)

# list of 1D tensors [ batch_size ] of length num_steps
loss_weights = tf.unstack(tf.cast(mask, tf.float32), axis=1)

# dropout
dropout = tf.placeholder(tf.float32, name="placeholder_dropout")

lstm = tf.contrib.rnn.BasicLSTMCell(
    hidden_size, forget_bias=1, state_is_tuple=True)
lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout)
stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    [lstm] * num_layers, state_is_tuple=True)

initial_state = stacked_lstm.zero_state(batch_size, dtype=tf.float32)

encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    cell=stacked_lstm, inputs=encoder_inputs, initial_state=initial_state, dtype=tf.float32, sequence_length=encoder_lengths)

decoder_outputs, decoder_state = tf.contrib.legacy_seq2seq.rnn_decoder(
    decoder_inputs=decoder_inputs, initial_state=encoder_state, cell=stacked_lstm)

preds = [tf.matmul(step, decoder_weights) +
         decoder_bias for step in decoder_outputs]

loss = tf.contrib.legacy_seq2seq.sequence_loss(
    logits=preds, targets=targets, weights=loss_weights)

# optimizer = tf.train.GradientDescentOptimizer(.01)
optimizer = tf.train.AdamOptimizer(.00001)
gradients = optimizer.compute_gradients(loss)
clipped_gradients = [(tf.clip_by_norm(grad, 5), var) for grad, var in gradients]
train_op = optimizer.apply_gradients(clipped_gradients)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

#############################

en_val_subset_file_path = "data/en_val_subset.gz"
fr_val_subset_file_path = "data/fr_val_subset.gz"

en_onehot_tok_idx = "data/en_onehot.npy"
fr_onehot_tok_idx = "data/fr_onehot.npy"

onehot_tok_idx = np.load(en_onehot_tok_idx).item()

batch_size = 384
num_steps = 48

val_iter_ = data_iterator([en_val_subset_file_path], onehot_tok_idx, 1, batch_size, num_steps)

total_val_samples = 3000

val_iterations = int(total_val_samples / batch_size)


with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "tmp/second_run/model_2604.ckpt")
    
    print("model loaded")

    validation_accuracy = 0

    for j in range(0, val_iterations):

        val_sequences_batch, val_labels_batch = val_iter_.__next__()

        validation_accuracy += loss.eval(session=sess, feed_dict={
                                   encoder_inputs: val_sequences_batch, raw_labels: val_labels_batch, dropout: 1.0})

    validation_accuracy /= val_iterations

    print("validation loss %g" % (validation_accuracy))

