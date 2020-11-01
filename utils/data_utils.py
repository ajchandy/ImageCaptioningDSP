from keras import vgg16
img_ids, caption

img_feat_dict = {}

PATH_OUT = ""

def get_img_feat(path, model='VGG'):

    for ids in img_ids:
        img = img_to_array(load_img(f"{path_to_img}/{ids}.jpg"))
        # size = (224,224,3) => (1, 224, 224, 3)
        img = np.expand_dims(preprocess_img(img), axis=0)

        # (1, 1, 1, 2048)
        feat_vec = vgg16(img)
        # size (1,1,1,2048) => (1,2048)
        feat_vec = np.reshape(feat_vec, (1, -1))
        img_feat_dict[ids] = feat_vec

    dump(img_feat_dict.)

    return PATH_OUT
