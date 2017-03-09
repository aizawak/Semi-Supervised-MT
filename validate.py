import tensorflow as tf
import numpy as np
from data_preprocess import data_iterator

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
    saver = tf.train.import_meta_graph("tmp/second_run/model_2604.ckpt.meta")
    saver.restore(sess, "tmp/second_run/model_2604.ckpt")
    
#    preds = [tf.matmul(step, decoder_weights) + decoder_bias for step in decoder_outputs]
#    loss = tf.contrib.legacy_seq2seq.sequence_loss(logits=preds, targets=targets, weights=loss_weights)
    mask = tf.sign(tf.reduce_max(tf.abs(encoder_inputs), axis=2))

    encoder_lengths = tf.cast(tf.reduce_sum(mask, reduction_indices=1), tf.int32)

    decoder_inputs = tf.unstack(tf.transpose(encoder_inputs, [1, 0, 2]), axis=0)
    decoder_inputs = ([tf.zeros_like(decoder_inputs[0], name="GO")] + decoder_inputs[:-1])

    targets = tf.unstack(raw_labels, axis=1)

    loss_weights = tf.unstack(tf.cast(mask, tf.float32), axis=1)

    print("model loaded")

    validation_accuracy = 0

    for j in range(0, val_iterations):

        val_sequences_batch, val_labels_batch = val_iter_.__next__()

        validation_accuracy += loss.eval(session=sess, feed_dict={
                                   encoder_inputs: val_sequences_batch, raw_labels: val_labels_batch, dropout: 1.0})

    validation_accuracy /= val_iterations

    print("validation loss %g" % (validation_accuracy))

