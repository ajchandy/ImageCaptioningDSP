import os

print(os.getcwd())
from keras.layers import Embedding, Dense, BatchNormalization, Dropout, LSTM, add
from keras.models import Model, Input
from mypkg.loaders.datagen import MAX_LENGTH, VOCAB_SIZE

EMBED_SIZE = 300


def comb_model(img_feat_dim=512, act="relu"):
    """Start defining the model"""
    # Img projection part
    imginp = Input(shape=(img_feat_dim,), name='Image_input')
    imglay1 = Dropout(0.5)(imginp)
    imglay2 = Dense(EMBED_SIZE, activation=act, name="Img_Proj_Dense")(imglay1)
    # LSTM Part
    textinp = Input(shape=(MAX_LENGTH,), name="Cap_Input")
    textlay1 = Embedding(VOCAB_SIZE, EMBED_SIZE, mask_zero=True, name="Embed_Layer")(
        textinp
    )
    textlay2 = Dropout(0.5)(textlay1)
    textlay3 = LSTM(EMBED_SIZE, name="LSTM_Layer")(textlay2)
    # Decoder part that combines both
    declay1 = add([imglay2, textlay3])
    declay2 = Dense(EMBED_SIZE, activation=act, name="Dec_Dense")(declay1)
    output = Dense(VOCAB_SIZE, activation="softmax", name="Output_Layer")(declay2)
    # Creating keras model
    model = Model(inputs=[imginp, textinp], outputs=output)
    return model


# Test model creation
if __name__ == "__main__":
    # VGG model
    mdl1 = comb_model()
    print(mdl1.summary())
    # Inception v3 model
    mdl2 = comb_model(img_feat_dim=2048)
    print(mdl2.summary())
