import re
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img,  img_to_array
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc
import numpy as np
from pickle import dump,load

#MODEL = VGG16(include_top=False, pooling='avg')
MODEL=InceptionV3(include_top=False,pooling='avg')
PATH_IN = "C:/Users/akhil/DSP/Flicker Data/Flickr8k_text/Flickr_8k.trainImages.txt"
PATH_OUT_Inc = "C:/Users/akhil/PycharmProjects/Image-captioning---DSP/pkl files"
DATA_PATH = "C:/Users/akhil/DSP/Flicker Data/Flickr8k_Dataset/Flicker8k_Dataset"
PRE_PROCESS = preprocess_input_inc

img_feat_dict = {}


def get_img_feat(path_in, data_path, path_out, model, pre_process):
    ''' Method to extract image feature vectors using CNN and
    dump as a pickle file'''

    TARGET_SIZE = (224, 224)

    with open(path_in)as f:
        train_img_data = f.read()

    for l in train_img_data.split('\n'):
        image_id = l.split('.')[0]
        # Loading individual images
        img = load_img(f"{data_path}/{image_id}.jpg", target_size=TARGET_SIZE)
        # Converting image to array
        img_array = img_to_array(img)
        # array shape - (224,224,3)
        n_img = pre_process(img_array)
        # Adding one more dimension to feed to model in batches
        n_img = np.expand_dims(n_img, axis=0)
        # Extracting features from image
        feat_vec = model.predict(n_img)
        # feat vec shape (1,512)
        img_feat_dict[image_id] = np.reshape(feat_vec, (1, -1))

    with open(path_out, 'wb') as f:
        dump(img_feat_dict, f)


def image_description(path_in, path_out):
    '''
    Method to create and dump a pickle file of image descriptions
    Takes two string inputs,
    path to token file and output for descriptions pickle file'''
    with open(path_in) as f:
        data = f.read()

    # define a dictionary to store images and descriptions
    image_desc = dict()

    # converts input text to an array of images and captions
    lines = data.split('\n')

    # create a loop to add image name as key name and captions for each image as a list value for each key element
    for l in lines:
        desc = l.split('\t')
        id , desc = desc[0], desc[1:]

        # Drop .jpg from the image id
        id = id.split('.')[0]

        desc = " ".join(desc)

        # check to see if image id exists in dictionary
        if id in image_desc:
            image_desc[id].append(desc)
        else:
            image_desc[id] = list()
            image_desc[id].append(desc)

    with open(f'{path_out}/image_descriptions.pkl', 'wb') as f:
        dump(image_desc, f)


"""
Write code to extract image features as train, test and split blocks
Also extract captions and make ready to be fed into LSTM

Data cleaning for text convert to lower case, remove special characters 
Create a corpus with all unique words in the captions
Step 1 Extract image features
Step 2 Clean up captions, create corpus
Step 3 Encode each caption with a start and stop seq token
Step 4 Map each word in corpus to a higher dimension vecto
Step 5 Train the LSTM
  
"""

def replace_contracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def append_seq(d, path_out)
'''
Appends start and stop sequence to data files
'''
    for l in d.keys():
        caption_list = d[l]
        new_caption = []
        for c in caption_list:
            # replace special characters in the sentences
            sent = replace_contracted(c)
            sent = sent.replace('\\r', ' ')
            sent = sent.replace('\\"', ' ')
            sent = sent.replace('\\n', ' ')
            sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

            append_seq = 'startseq ' + sent.lower() + 'endseq'
            new_caption.append(append_seq)
        d[l] = new_caption

    with open(f'{path_out}/image_description_seq_append.pkl', 'wb') as f:
        dump(d, f)


def image_name_extract(description, p):
    '''
    Extracts image names looking up names from the text file and image_description_seq_append dictionary
    Returns a dict of image names with captions

    '''
    image_descriptions = {}

    with open(p, 'r') as f:
        data = f.read()

    for lines in data.split('\n'):
        captions = []
        image_id = lines.split('.')[0]
        image_descriptions[image_id] = description[image_id]
    return image_descriptions


def description_split(description, path_in, path_out):
    '''
    Dumps data into different pickle files for train,test and dev captions
    Takes image_description_seq_append dictionary, input and output path as arguments

    '''
    if path_in.split('/')[6][10:] == 'trainImages.txt':
        train_image_captions = image_name_extract(description, path_in)

        # Dumps data
        with open(f'{path_out}/train_image_descriptions.pkl', 'wb') as f:
            dump(train_image_captions, f)

    if path_in.split('/')[6][10:] == 'testImages.txt':
        test_image_captions = image_name_extract(description, path_in)

        # Dumps data
        with open(f'{path_out}/test_image_descriptions.pkl', 'wb') as f:
            dump(test_image_captions, f)

    if path_in.split('/')[6][10:] == 'devImages.txt':
        dev_image_captions = image_name_extract(description, path_in)

        # Dumps data
        with open(f'{path_out}/dev_image_descriptions.pkl', 'wb') as f:
            dump(dev_image_captions, f)
