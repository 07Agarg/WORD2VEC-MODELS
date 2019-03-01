# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 07:06:16 2018

@author: ashima.garg
"""
import config
import os
import numpy as np
import collections 
import zipfile
from six.moves.urllib.request import urlretrieve

class DATA():

    def __init__(self):
        self.batch = None
        self.data = None
        self.size = None
        self.dictionary = None
        self.count = None
        self.reverse_dictionary = None
        self.num_skips = config.NUM_SKIPS
        self.skip_window = config.SKIP_WINDOW
        self.data_index = 0
        self.batch_size = config.BATCH_SIZE
        self.vocab_size = config.VOCABULARY_SIZE

    def maybe_download(filename, expected_bytes):
        if not os.path.exists(filename):
            filename, _ = urlretrieve(config.url + filename, filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified %s' % filename)
        else:
            print(statinfo.st_size)
            raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
        return filename
    
    def read_data(self, filename):
        with zipfile.ZipFile(filename) as f:
            data = f.read(f.namelist()[0]).decode('utf-8').split()
        return data
    
    def build_dataset(self, filename):
        words = self.read_data(filename)
        self.count = [['UNK', -1]]
        self.count.extend(collections.Counter(words).most_common(self.vocab_size - 1))
        self.dictionary = dict()
        self.data = list()
        unk_count = 0
        for word, _ in self.count:
            self.dictionary[word] = len(self.dictionary)
        for word in words:
            index = self.dictionary.get(word, 0)
            if index == 0:  # dictionary['UNK']
                unk_count += 1
            self.data.append(index)
        self.count[0][1] = unk_count
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
    
    def generate_batch(self):
        span = 2 * self.skip_window + 1
        batch = np.ndarray(shape = (self.batch_size, span - 1), dtype = np.int32)
        labels = np.ndarray(shape = (self.batch_size, 1), dtype = np.int32)
        buffer = collections.deque(maxlen = span)
        if self.data_index + span > len(self.data):
            self.data_index = 0
        buffer.extend(self.data[self.data_index: self.data_index + span])
        self.data_index += span
        for i in range(self.batch_size):
            context_words = [w for w in range(span) if w != self.skip_window]
            col_idx = 0
            for j, context_word in enumerate(context_words):
                if j == span // 2:
                    continue
                batch[i, col_idx] = buffer[j]
                col_idx += 1
            labels[i, 0] = buffer[self.skip_window]
            if self.data_index == len(self.data):
                buffer[:] = self.data[:span]
                self.data_index = span
            else:
                buffer.append(self.data[self.data_index])
                self.data_index += 1
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels