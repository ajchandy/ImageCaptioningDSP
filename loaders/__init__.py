#!python3

"""
This subdir is meant for having a efficient
DataLoaders to load the images/labels

USAGE:
Create new *.py under this subdir

The file should contain a pyfunc:

Parameters
-------------
data_path  : Path to the data files
batch_size : Batch_size for the dataloaders
**kwargs (any other required args)

Returns
-------------
train_loader : A Dataloader object that gives batches of train images & labels
validation_loader : A Dataloader object that gives batches of Validation set
test_loader : A Dataloader object that gives batches of Test Set
"""
