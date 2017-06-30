import score
import rnn
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


score.score(rnn.rnn)

