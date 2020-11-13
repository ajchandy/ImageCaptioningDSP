import os
import pickle

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

VOCAB_SIZE = 1706
MAX_LENGTH = 39


def train_loader(img_desc, img_feat, token, bsize):

    # img_desc = pickle.load(open(desc_path,'r'))
    # img_feat = pickle.load(open(img_feat_path,'r'))
    # token = pickle.load(token_path)
    inp1, inp2, label = [], [], []
    cnt = 0
    # while 1:
    for k, desc in img_desc.items():
        cnt += 1
        ifeat = np.reshape(img_feat[k], (img_feat[k].shape[-1],))
        for d in desc:
            seq = token.texts_to_sequences([d])
            seq = seq[0]
            for i in range(1, len(seq)):
                inpseq, lbl = seq[:i], seq[0]
                inpseq = pad_sequences([inpseq], maxlen=MAX_LENGTH, padding="post")[0]
                lbl = to_categorical([lbl], num_classes=VOCAB_SIZE)[0]
                inp1.append(ifeat)
                inp2.append(inpseq)
                label.append(lbl)
        if cnt == bsize:
            yield [[np.array(inp1), np.array(inp2)], np.array(label)]
            inp1, inp2, label = [], [], []
            cnt = 0


# Check the Loader
if __name__ == "__main__":
    # Test the loader
    path = "/Users/prateethvnayak/Documents/Agile/Image-captioning---DSP/pkl_files"
    img_desc = pickle.load(open(f"{path}/train_image_descriptions.pkl", "rb"))
    img_feat = pickle.load(open(f"{path}/train_image_extracted_VGG16.pkl", "rb"))
    token = pickle.load(open(f"{path}/token.pkl", "rb"))
    dl = train_loader(img_desc, img_feat, token, bsize=10)
    for k in dl:
        print(k[0][0].shape, k[0][1].shape, k[-1].shape)
        break
