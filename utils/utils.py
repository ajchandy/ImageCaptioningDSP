import os
import argparse


def get_args():
    """
    Argparser for arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", dest="train", store_action=True, default=False)
    parser.add_argument("-backbone", dest="bckb", type=str, default="vgg16")
    parser.add_argument("-eval", dest="eval", store_action=True, default=False)
    parser.add_argument("-data", dest="datapath", type=str, default="pkl files/")
    parser.add_argument("-optim", dest="optim", type=str, default="adam")

    args = parser.parse_args()
    return args