from nltk import FreqDist
from pickle import load,dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np

# creating corpus
corpus = ""

with open("pkl files/train_image_descriptions.pkl","rb") as f:
    train_descriptions = load(f)
for ec in train_descriptions.values():
    for el in ec:
        corpus += " "+el

''' All sentences in the text file added to corpus'''
total_words = corpus.split()
vocabulary = set(total_words)
print("The size of vocablury is {}".format(len(vocabulary)))


# creating frequency distribution of words
freq_dist = FreqDist(total_words)
print(freq_dist.most_common(10))


#removing least common words from vocabulary
for ew in list(vocabulary):
    if(freq_dist[ew]<10):
        vocabulary.remove(ew)


VOCAB_SIZE = len(vocabulary)+1
print("Total unique words after remooving less frequent word from our corpus = {}".format(VOCAB_SIZE))


caption_list = []
for el in train_descriptions.values():
    for ec in el:
        caption_list.append(ec)
print("The total caption present = {}".format(len(caption_list)))


token = Tokenizer(num_words=VOCAB_SIZE)
token.fit_on_texts(caption_list)


# index to words are assigned according to frequency. i.e the most frequent word has index of 1
ix_to_word = token.index_word

for k in list(ix_to_word):
    if k>=1665:
        ix_to_word.pop(k, None)


word_to_ix = dict()
for k,v in ix_to_word.items():
    word_to_ix[v] = k


print(len(word_to_ix))
print(len(ix_to_word))

# finding the max_length caption
MAX_LENGTH = 0
temp = 0
for ec in caption_list:
    temp = len(ec.split())
    if(MAX_LENGTH<=temp):
        MAX_LENGTH = temp


print("Maximum caption has length of {}".format(MAX_LENGTH))

# make sure you have the glove_vectors file
with open('pkl files/glove_vector.pkl', 'rb') as f:
    glove = load(f)
    glove_words = set(glove.keys())


EMBEDDING_SIZE = 300

# Get 300-dim dense vector for each of the words in vocabulary
embedding_matrix = np.zeros((VOCAB_SIZE,EMBEDDING_SIZE))
print(embedding_matrix.shape)

# Get 300-dim dense vector for each of the words in vocabulary
embedding_matrix = np.zeros(((VOCAB_SIZE),EMBEDDING_SIZE))

for word, i in word_to_ix.items():
    embedding_vector = np.zeros(300)
    if word in glove_words:
        embedding_vector = glove[word]
        embedding_matrix[i] = embedding_vector
    else:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector


# save the embedding matrix to file
with open("C:/Users/akhil/PycharmProjects/Image-captioning---DSP/pkl files/embedding_matrix.pkl","wb") as f:
    dump(embedding_matrix,f)

print(embedding_matrix.shape)


# save the tokenizer to file
with open("C:/Users/akhil/PycharmProjects/Image-captioning---DSP/pkl files/token.pkl","wb") as f:
    dump(token,f)

