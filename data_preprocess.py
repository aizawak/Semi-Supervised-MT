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


def read_data(file_paths, write_path, num_batches, top_n):

    tok_counts = collections.Counter()

    for file_path in file_paths:

        print("...processing file: %s" % file_path)

        with gzip.open(file_path, 'rb') as f:
            content = f.read()

        for i in range(0, num_batches):
            start = math.floor(i * len(content) / num_batches)
            end = math.floor((i + 1) * len(content) / num_batches)

            toks = re.findall(r"[\w]+|[^\s\w]",
                              content[start:end].decode('utf-8'))

            for tok in toks:
                tok_counts[tok] += 1

            del toks

        print("...tokens counted for file: %s" % file_path)

    del content

    tok_counts = tok_counts.most_common(top_n - 1)

    onehot_tok_idx = {"UNK": 0}

    tok_idx = 1

    for tok, count in tok_counts:

        if tok not in onehot_tok_idx:
            onehot_tok_idx[tok] = tok_idx
            tok_idx += 1

    np.save(write_path, onehot_tok_idx)
    print("...token id's saved")


def generate_val(file_path, write_path, num_val):

    with gzip.open(file_path, 'rb') as f:
        content = f.read().decode('utf-8').split('\n')

    np.random.seed(seed=1)

    np.random.shuffle(content)

    raw = bytearray("\n".join(content[0:num_val]), "utf-8")

    with gzip.open(write_path, 'wb') as wf:
        wf.write(raw)

    print("...validation set saved")

# Loop through batches and generate one-hot encodings of sequences and
# sequence labels.

# file_path = subtitles.gz file path
# onehot_tok_idx = dictionary of token id's
# num_batches = number of batches to split data parsing
# batch_size = size of batches to generate
# seq_length = max sequence length


def data_iterator(file_paths, onehot_tok_idx, num_batches, batch_size, seq_length):

    while True:

        for file_path in file_paths:

            with gzip.open(file_path, 'rb') as f:
                content = f.read()

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
                                tok = "UNK"

                            onehot_seq_batch[batch_idx][
                                tok_idx][onehot_tok_idx[tok]] = 1
                            labels_batch[batch_idx][
                                tok_idx] = onehot_tok_idx[tok]
                        yield onehot_seq_batch, labels_batch

                del sent

if __name__ == "__main__":

    en_file_sources = ["http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.en.shuffled.gz", "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.en.shuffled.gz", "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.en.shuffled.gz",
                       "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.en.shuffled.gz", "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.en.shuffled.gz", "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz", "http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.en.shuffled.v2.gz"]
    fr_file_sources = ["http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.fr.shuffled.gz", "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.fr.shuffled.gz", "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.fr.shuffled.gz",
                       "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.fr.shuffled.gz", "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.fr.shuffled.gz", "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.fr.shuffled.gz", "http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.fr.shuffled.v2.gz"]

    en_file_paths = ["data/en_%d.gz" %
                     i for i in range(0, len(en_file_sources))]
    fr_file_paths = ["data/fr_%d.gz" %
                     i for i in range(0, len(fr_file_sources))]

    en_val_file_source = "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.en.shuffled.gz"
    fr_val_file_source = "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.fr.shuffled.gz"

    en_val_file_path = "data/en_val.gz"
    fr_val_file_path = "data/fr_val.gz"

    en_val_write_path = "data/en_val_subset.gz"
    fr_val_write_path = "data/fr_val_subset.gz"

    en_write_path = "data/en_onehot.npy"
    fr_write_path = "data/fr_onehot.npy"

    top_n = 40000

    print("processing english files")

    for i in range(0, len(en_file_sources)):

        file_source = en_file_sources[i]
        file_path = en_file_paths[i]

        if not os.path.isfile(file_path):
            urllib.request.urlretrieve(
                file_source, filename=file_path)

    if not os.path.isfile(en_val_file_path):
        urllib.request.urlretrieve(
            en_val_file_source, filename=en_val_file_path)

    print("...files downloaded")

    read_data(file_paths=[en_file_paths[6]], write_path=en_write_path,
              num_batches=1, top_n=top_n)

    generate_val(en_val_file_path, en_val_write_path, 3000)

    print("processing french files")

    for i in range(0, len(fr_file_sources)):

        file_source = fr_file_sources[i]
        file_path = fr_file_paths[i]

        if not os.path.isfile(file_path):
            urllib.request.urlretrieve(
                file_source, filename=file_path)

    if not os.path.isfile(fr_val_file_path):
        urllib.request.urlretrieve(
            fr_val_file_source, filename=fr_val_file_path)

    print("...files downloaded")

    read_data(file_paths=[fr_file_paths[6]], write_path=fr_write_path,
              num_batches=1, top_n=top_n)

    generate_val(fr_val_file_path, fr_val_write_path, 3000)
