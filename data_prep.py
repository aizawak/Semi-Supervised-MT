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

import os
from random import shuffle
import re
import collections
import urllib.request
import gzip
import math

def generate_data(file_path, num_batches, seq_length):
    
    onehot_tok_idx = {}
    tok_idx_sent = np.empty(shape=(len(sent), seq_length), dtype="int32")

    tok_idx=1

    for i in range(0, num_batches):
        
        with gzip.open(file_path, 'rb') as f:
            content = f.read()
        
        start = math.floor(i * len(english_content) / num_batches)
        end = math.floor((i + 1) * len(english_content) / num_batches)

        print("...subtitles in memory for batch %d/%d"%(i+1, num_batches))

        sent = content.split('\n')
        del content

        seq_idx=seq_length-1

        for sent_idx in range(len(sent)-1, -1):

            sent_tok = re.findall(r"[\w]+|[^\s\w]", sent[sent_idx])

            for tok in sent_tok:
                if seq_idx < 0:
                    break

                if tok not in onehot_tok_idx_en:
                    onehot_tok_idx[tok]=tok_idx
                    tok_idx+=1

                tok_idx_sent[sent_idx][seq_idx]=onehot_tok_idx[tok]

                seq_idx-=1

        del sent

    print("...subtitles cleaned, token id's assigned for batch %d/%d"%(i+1,num_batches))

    return (onehot_tok_idx, tok_idx_sent)

en_file_path = "data/english_subtitles.gz"
fr_file_path = "data/french_subtitles.gz"

en_onehot_path = "data/en_onehot.npy"
fr_onehot_path = "data/fr_onehot.npy"

en_tok_idx_sent_path = "data/en_tok_idx_sent.npy"
fr_tok_idx_sent_path = "data/fr_tok_idx_sent.npy"

num_batches = 5
seq_length = 100
tok_idx = 1

print("processing english subtitles")

if not os.path.isfile(en_file_path):
    urllib.request.urlretrieve("http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/mono/OpenSubtitles2016.raw.en.gz", filename=en_file_path)

print("...subtitles downloaded")

out = generate_data(en_file_path, num_batches, seq_length)

np.save(en_onehot_path, out[0])

print("...token id's saved")

np.save(en_tok_idx_sent_path, out[1])

print("...subtitles saved")

del out

print("processing french subtitles")

if not os.path.isfile(fr_file_path):
    urllib.request.urlretrieve("http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/mono/OpenSubtitles2016.raw.fr.gz", filename=fr_file_path)
    
print("...subtitles downloaded")

out = generate_data(fr_file_path, num_batches, seq_length)

np.save(fr_onehot_path, out[0])

print("...token id's saved")

np.save(fr_tok_idx_sent_path, out[1])

print("...subtitles saved")

del out