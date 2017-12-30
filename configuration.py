import os
import datetime
import numpy as np
import tensorflow as tf

## These lines ensure any existing tf sessions are closed.
try:
    tf.Session().close()
except:
    pass

## Numpy print options
np.set_printoptions(precision=3)

## Directories 
experiment_directory = os.environ["PL_EXPERIMENTS"]  # contains experiment yaml files
data_directory = os.environ["PL_DATA"]  # contains cPickle data files 
output_directory = os.environ["PL_OUT"]  # contains experiment results

## Random Seeds
# each random seed represents an experimental replication.
# You can add or remove list elements to change the number
# of replications for an experiment.
seeds = [
    {"tf_seed": 649737, "np_seed": 29820},
    {"tf_seed": 395408, "np_seed": 185228},
    {"tf_seed": 252356, "np_seed": 703889},
    {"tf_seed": 343053, "np_seed": 999360},
    {"tf_seed": 743746, "np_seed": 67440},
    {"tf_seed": 175343, "np_seed": 378945},
    {"tf_seed": 856516, "np_seed": 597688},
    {"tf_seed": 474313, "np_seed": 349903},
    {"tf_seed": 838382, "np_seed": 897904},
    {"tf_seed": 202003, "np_seed": 656146},
]


# A slightly fancy printing method
def printt(msg):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("{}| {}".format(time_str, msg))
