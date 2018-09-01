# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 07:06:16 2018

@author: ashima.garg
"""
import os

# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/')

TRAIN_FILE = "text8.zip"
RESULT_FILE = "tsne_plot.png"
url = 'http://mattmahoney.net/dc/'

BYTE_SIZE = 31344016
# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

VOCABULARY_SIZE = 50000
EMBEDDING_SIZE = 128
BATCH_SIZE = 128
SKIP_WINDOW = 1
NUM_SKIPS = 2             # of times we select a random word within the span

NUM_STEPS = 100001
NUM_SAMPLED = 64