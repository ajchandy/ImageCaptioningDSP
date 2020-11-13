#!python3

"""
This subdir is meant for the implementing
the model arch

Proposed Pipeline :

VGG16 --> Transformer

This should have a pyfunc:

Parameters:
-------------
preTrain_model : A path-to/load-directly a pretrained model
**kwargs [can be a arg-less pyfunc since we fix only on one arch]

Returns:
-------------
Model : A Keras/TF module that encapsulates the whole architecture.

"""
