## imports
import sys
import os

if os.path.exists('../../pynncml'):
    print("Import PyNNCML From Code")
    # sys.path.append('../../')  # This line is need to import pynncml
    sys.path.insert(0, '../../')
else:
    print("Install PyNNCML From pip")
    # !pip install pynncml

import numpy as np
import pynncml as pnc
import torch
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy
from sklearn import metrics


## parameters
#  @title Hyper-parameters
batch_size = 16  # @param{type:"integer"}
window_size = 32  # @param{type:"integer"}
rnn_n_features = 128  # @param{type:"integer"}
metadata_n_features = 32  # @param{type:"integer"}
n_layers = 2  # @param{type:"integer"}
lr = 1e-4  # @param{type:"number"}
weight_decay = 1e-4  # @param{type:"number"}
rnn_type = pnc.neural_networks.RNNType.GRU  # RNN Type
n_epochs = 3 #200  # @param{type:"integer"}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained = False

# print current working directory
print("Current working directory:\n", os.getcwd())


## build dataset
xy_min = [1.29e6, 0.565e6]  # Link Region
xy_max = [1.34e6, 0.5875e6]
time_slice = slice("2015-06-01", "2015-06-10")  # Time Interval
dataset = pnc.datasets.loader_open_mrg_dataset(data_path=".\\data\\", xy_min=xy_min, xy_max=xy_max, time_slice=time_slice)

example_link = dataset[0]
print(example_link)

# dataset.link_set.plot_links(scale=True, scale_factor=1.0)
# plt.grid()
# plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
# plt.show()

# split dataset to training and validation
training_dataset, validation_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
data_loader = torch.utils.data.DataLoader(training_dataset, batch_size)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size)

model = pnc.scm.rain_estimation.two_step_network(n_layers=n_layers,  # Number of RNN layers
                                                 rnn_type=rnn_type,  # Type of RNN (GRU, LSTM)
                                                 normalization_cfg=pnc.training_helpers.compute_data_normalization(
                                                     data_loader),
                                                 # Compute the normalization statistics from the training dataset.
                                                 rnn_input_size=180,  # 90 + 90 (RSL + TSL)
                                                 rnn_n_features=rnn_n_features,  # Number of features in the RNN
                                                 metadata_input_size=len(dataset[0][-1]), #2,  # Number of metadata features
                                                 metadata_n_features=metadata_n_features,
                                                 # Number of features in the metadata
                                                 pretrained=pretrained).to(device)  # Pretrained model is set to False to train the model from scratch.

print(f"{model.bb.fc_meta.weight.shape = }")
