'''
This code ties up the caption for each image with its image ID in a dictionary
The image captions are present in the token.txt file
'''
PATH='C:/Users/akhil/DSP/Flicker Data/Flickr8k_text/Flickr8k.token.txt'

def dataloader(path):
    """
    :param path: Path to data file
    :return: image_desc : Dict image id and caption
    """
    with open(path) as f:
        data=f.read()

    image_desc=dict()

    lines = data.split('\n')

    for l in lines:
        desc = l.split('\t')
    id, desc = desc[0], desc[1:]
    # Drop .jpg from the image id
    id = id.split('.')[0]

    desc = " ".join(desc)

    # check to see if image id exists in dictionary
    if id in image_desc:
        image_desc[id].append(desc)
    else:
        image_desc[id] = list()
        image_desc[id].append(desc)

    return image_desc


