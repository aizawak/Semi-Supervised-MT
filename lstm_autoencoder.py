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

vocab_size = len(onehot_tok_idx)
num_layers = 8
num_steps = 100
batch_size = 50
hidden_size = 2000

seq_input = tf.placeholder(
    tf.int32, [num_steps, batch_size, vocab_size], name="sequence_placeholder")

encoder_inputs = [tf.reshape(seq_input, [-1, vocab_size])]

labels = [tf.reshape(seq_input, [-1, vocab_size])]

weights = [tf.ones_like(label, dtype=tf.float16) for label in labels]

decoder_inputs = (
    [tf.zeros_like(encoder_inputs[0], name="GO")] + encoder_inputs[:-1])

lstm = tf.contrib.rnn.BasicLSTMCell(
    hidden_size, forget_bias=0, state_is_tuple=True)
stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    [lstm] * num_layers, state_is_tuple=True)

outputs, state = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(
    encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs, cell=stacked_lstm, dtype=tf.float16)

# outputs_test, state_test = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(
#    encoder_inputs, decoder_inputs, stacked_lstm, feed_previous=True)

loss = tf.contrib.legacy_seq2seq.sequence_loss(
    outputs, labels, weights, vocab_size)

optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss)

iter_ = data_iterator(fr_file_path, onehot_tok_idx,batch_size, num_steps)

saver = tf.train.Saver()

sess.run(tf.initialize_all_variables())

with tf.Session() as sess:
    sess.run(init)
    for i in range(200000):
        sequences_batch = iter_.__next__()

        if (i+1)%100==0:
            train_accuracy = loss.eval(session = sess, feed_dict={encoder_inputs: sequences_batch, labels: sequences_batch})
            print("step %d, training loss %g"%(i+1, train_accuracy))

        optimizer.run(session = sess, feed_dict={seq_input: sequences_batch})



