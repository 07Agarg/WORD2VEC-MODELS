# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 07:06:16 2018

@author: ashima.garg
"""


import config
import data
import model
import utils
import os


if __name__ == "__main__":
    # LOAD DATA
    train_data = data.DATA()
    train_data.build_dataset(os.path.join(config.DATA_DIR, config.TRAIN_FILE))
    print("Train data Loaded")
    #BUILD MODEL
    model = model.MODEL()
    model.build()
    print("model built")
    #TRAIN MODEL
    model.train(train_data)
    print("Model Trained")
    utils.plot_with_labels(model.embeddings, train_data)
    
