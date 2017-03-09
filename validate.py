import tensorflow as tf
from data_preprocess import data_iterator

en_val_subset_file_path = "data/en_val_subset.gz"
fr_val_subset_file_path = "data/fr_val_subset.gz"

en_onehot_tok_idx = "data/en_onehot.npy"
fr_onehot_tok_idx = "data/fr_onehot.npy"

onehot_tok_idx = np.load(en_onehot_tok_idx).item()

batch_size = 384
num_steps = 48

val_iter_ = data_iterator([en_val_subset_file_path], onehot_tok_idx, 1, batch_size, num_steps)

saver = tf.train.Saver()

total_val_samples = 3000

val_iterations = int(total_val_samples / batch_size)


with tf.Session() as sess:
	saver.restore(sess, "/tmp/second_run/model_2604.ckpt")

	print("model loaded")

	validation_accuracy = 0

    for j in range(0, val_iterations):

        val_sequences_batch, val_labels_batch = val_iter_.__next__()

        validation_accuracy += loss.eval(session=sess, feed_dict={
                                   encoder_inputs: val_sequences_batch, raw_labels: val_labels_batch, dropout: 1.0})

    validation_accuracy /= val_iterations

    print("validation loss %g" % (validation_accuracy))

