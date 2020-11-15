import os

print(os.getcwd())
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Dropout, LSTM, add
from tensorflow.keras import Model, Input
from mypkg.loaders.datagen import MAX_LENGTH, VOCAB_SIZE
from keras.models import model_from_json
from keras.models import load_model


EMBED_SIZE = 300


def comb_model(embed_mat,  name, img_feat_dim=512, act="relu",):
    """Start defining the model"""
    # Img projection part
    imginp = Input(shape=(512,))
    imglay1 = Dropout(0.5)(imginp)
    imglay2 = Dense(EMBED_SIZE, activation=act)(imglay1)
    # LSTM Part
    textinp = Input(shape=(39,))
    textlay1 = Embedding(VOCAB_SIZE, EMBED_SIZE, mask_zero=True)(
        textinp
    )
    textlay2 = Dropout(0.5)(textlay1)
    textlay3 = LSTM(EMBED_SIZE)(textlay2)
    # Decoder part that combines both
    declay1 = add([imglay2, textlay3])
    declay2 = Dense(EMBED_SIZE, activation=act)(declay1)
    output = Dense(VOCAB_SIZE, activation="softmax")(declay2)
    # Creating keras model
    model = Model(inputs=({'input1': imginp, 'input2': textinp}), outputs=output)

    return model

# EXP
    # print(model.layers[2])
    # model.layers[2].set_weights([embed_mat])
    # model.layers[2].trainable = False

    # # model.compile(loss='categorical_crossentropy', optimizer='adam')

    # model_json = model.to_json()
    # with open(f"{os.getcwd()}/pkl_files/{name}model.json",'w') as json_file:
    #     json_file.write(model_json)

    # model.save_weights(f"{os.getcwd()}/pkl_files/{name}model.h5")


# Test model creation
if __name__ == "__main__":
    # VGG model
    import pickle
    emb = pickle.load(open(f"{os.getcwd()}/pkl_files/embedding_matrix.pkl",'rb'))
    mdl1 = comb_model(embed_mat=emb, name='vgg16')
    print(mdl1.summary())
    # Inception v3 model
    mdl2 = comb_model(embed_mat=emb,name='inceptionv3',img_feat_dim=2048)
    print(mdl2.summary())
