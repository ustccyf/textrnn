#using coding=utf8
import sys,os
from config import Config
import tensorflow as tf
import numpy as np
import logging
from rnn import rnn


if __name__ == "__main__":
    config = Config()
    train = config.train
    dev = config.dev
    rnn_model = rnn(config)
    rnn_model.build()    

