# Wiki Data Dump
# https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Amazon Product Reviews
# https://snap.stanford.edu/data/web-Amazon.htmls

# Book Data
# English Books
# http://opus.lingfil.uu.se/download.php?f=Books/en.tar.gz
# French Books
# http://opus.lingfil.uu.se/download.php?f=Books/fr.tar.gz

# English movie subtitles
# http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/mono/OpenSubtitles2016.raw.en.gz

# French movie subtitles
# http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/mono/OpenSubtitles2016.raw.fr.gz

import numpy as np
import os
from random import shuffle
import re
import collections
import urllib.request
import gzip
import math

# Count tokens and assign id's for each unique token. Use token counts to
# exclude rare tokens. Save one-hot token id's.

# file_path = subtitles.gz file path
# write_path = write path for dictionary of token id's
# num_batches = number of batches to split data parsing
# min_count = min number of tokens


def read_data(file_path, write_path, num_batches, top_n):

    with gzip.open(file_path, 'rb') as f:
        content = f.read()

    tok_counts = collections.Counter()

    for i in range(0, num_batches):
        start = math.floor(i * len(content) / num_batches)
        end = math.floor((i + 1) * len(content) / num_batches)

        toks = re.findall(r"[\w]+|[^\s\w]", content[start:end].decode('utf-8'))

        for tok in toks:
            tok_counts[tok] += 1

        del toks

    print("...tokens counted")

    tok_counts = tok_counts.most_common(top_n)

    onehot_tok_idx = {}

    tok_idx = 0

    for i in range(0, num_batches):

        start = math.floor(i * len(content) / num_batches)
        end = math.floor((i + 1) * len(content) / num_batches)

        sent = content[start:end].decode('utf-8').split('\n')

        print("...subtitles in memory for batch %d/%d" % (i + 1, num_batches))

        for sent_idx in range(0, len(sent)):

            sent_tok = re.findall(r"[\w]+|[^\s\w]", sent[sent_idx])

            for tok in sent_tok:
                if tok not in tok_counts:
                    tok = "UNKNOWNTEXT"
                if tok not in onehot_tok_idx:
                    onehot_tok_idx[tok] = tok_idx
                    tok_idx += 1

        del sent

        print("...token id's assigned for batch %d/%d" % (i + 1, num_batches))

    np.save(write_path, onehot_tok_idx)
    print("...token id's saved")

# Loop through batches and generate one-hot encodings of sequences and
# sequence labels.

# file_path = subtitles.gz file path
# onehot_tok_idx = dictionary of token id's
# num_batches = number of batches to split data parsing
# batch_size = size of batches to generate
# seq_length = max sequence length


def data_iterator(file_path, onehot_tok_idx, num_batches, batch_size, seq_length):

    with gzip.open(file_path, 'rb') as f:
        content = f.read()

    while True:
        for i in range(0, num_batches):

            start = math.floor(i * len(content) / num_batches)
            end = math.floor((i + 1) * len(content) / num_batches)

            sent = content[start:end].decode('utf-8').split('\n')

            onehot_seq_batch = np.zeros(
                shape=(batch_size, seq_length, len(onehot_tok_idx)), dtype="float32")
            labels_batch = np.zeros(
                shape=(batch_size, seq_length), dtype="int32")

            for batch_idx in range(0, batch_size):

                for sent_idx in range(batch_idx * batch_size, (batch_idx + 1) * batch_size):
                    if sent_idx >= len(sent):
                        break

                    sent_tok = re.findall(r"[\w]+|[^\s\w]", sent[sent_idx])

                    for tok_idx in range(0, len(sent_tok)):

                        if tok_idx >= seq_length:
                            break

                        tok = sent_tok[tok_idx]

                        if tok not in onehot_tok_idx:
                            tok = "UNKNOWNTEXT"

                        onehot_seq_batch[batch_idx][
                            tok_idx][onehot_tok_idx[tok]] = 1
                        labels_batch[batch_idx][tok_idx] = onehot_tok_idx[tok]
                    yield onehot_seq_batch, labels_batch

            del sent

if __name__ == "__main__":

    en_file_path = "data/english_subtitles.gz"
    fr_file_path = "data/french_subtitles.gz"

    en_write_path = "data/en_onehot.npy"
    fr_write_path = "data/fr_onehot.npy"

    top_n = 80000

   print("processing english subtitles")

   if not os.path.isfile(en_file_path):
       urllib.request.urlretrieve("http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/mono/OpenSubtitles2016.raw.en.gz", filename=en_file_path)

   print("...subtitles downloaded")

	read_data(file_path = en_file_path, write_path = en_write_path, num_batches = 5, top_n = top_n)

    print("processing french subtitles")

    if not os.path.isfile(fr_file_path):
        urllib.request.urlretrieve(
            "http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/mono/OpenSubtitles2016.raw.fr.gz", filename=fr_file_path)

    print("...subtitles downloaded")

    read_data(file_path = fr_file_path, write_path = fr_write_path, num_batches = 5, top_n = top_n)
