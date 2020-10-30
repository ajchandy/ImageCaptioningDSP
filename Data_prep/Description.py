'''
This code ties up the caption for each image with its image ID
The image captions are present in the token.txt file
'''

with open('C:/Users/akhil/DSP/Flicker Data/Flickr8k_text') as f:
    data=f.read()

image_desc=dict()


P