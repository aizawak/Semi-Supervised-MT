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

print("processing english subtitles")

if not os.path.isfile("data/english_subtitles.gz"):
    urllib.request.urlretrieve("http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/mono/OpenSubtitles2016.raw.en.gz", filename="data/english_subtitles.gz")

print("...subtitles downloaded")

num_batches = 5

onehot_tok_idx_en = {}
tok_idx_sent_en = []

tok_idx = 0

for i in range(0,chunks):

    with gzip.open('data/english_subtitles.gz', 'rb') as f:
        english_content = f.read()

    start = int(i * len(english_content) / num_batches)
    end = int((i + 1) * len(english_content) / num_batches)

    english_content = english_content[start:end].decode('utf-8')

    print("...subtitles in memory")

    sent_en = english_content.split('\n')
    del english_content

    for sent_idx in range(0, len(sent_en)):
        sent_tok = re.findall(r"[\w]+|[^\s\w]", sent_en[sent_idx])
        for tok in sent_tok:
            if tok not in onehot_tok_idx_en:
                onehot_tok_idx_en[tok]=tok_idx
                tok_idx+=1
        tok_idx_sent_en.append([onehot_tok_idx_en[tok] for tok in sent_tok])

    del sent_en

    print("...subtitles cleaned, token id's assigned for batch %d/%d"(i+1,num_batches))

np.save('data/onehot_tok_idx_en.npy', onehot_tok_idx_en)

print("...token id's saved")

np.save('data/tok_idx_sent_en.npy', tok_idx_sent_en)

print("...subtitles saved")



print("processing french subtitles")

if not os.path.isfile("data/french_subtitles.gz"):
    urllib.request.urlretrieve("http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/mono/OpenSubtitles2016.raw.fr.gz", filename="data/french_subtitles.gz")
    
print("...subtitles downloaded")

with gzip.open('data/french_subtitles.gz', 'rb') as f:
    french_content = f.read().decode('utf-8')

print("...subtitles in memory")

sent_fr = french_content.split('\n')
del french_content

onehot_tok_idx_fr = {}
tok_idx_sent_en = []

tok_idx = 0

for sent_idx in range(0, len(sent_fr)):
    sent_tok = re.findall(r"[\w]+|[^\s\w]", sent_fr[sent_idx])
    for tok in sent_tok:
        if tok not in onehot_tok_idx_fr:
            onehot_tok_idx_fr[tok]=tok_idx
            tok_idx+=1
    tok_idx_sent_fr.append([onehot_tok_idx_fr[tok] for tok in sent_tok])

del sent_fr

print("...subtitles cleaned, token id's assigned")

np.save('data/onehot_tok_idx_fr.npy', onehot_tok_idx_fr)

print("...token id's saved")

np.save('data/tok_idx_sent_fr.npy', tok_idx_sent_fr)

print("...subtitles saved")