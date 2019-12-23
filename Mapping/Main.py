# Coded by Roshan Gopalakrishnan, contact: roshan.gopalakrishnan@gmail.com
# Used for calculation of number of cores

from Mapping_functions import architecture_params
from Mapping_functions import core_utilization_per_layer

import sys
import os
import numpy as np
from platform import python_version


if python_version()[0] == str(3):
    print('Python version:', python_version())
else:
    print('Python version:', python_version())
    raise Exception('Python version: {} might have issues; use python 3'.format(python_version()))

# Parameters
architecture_path = "./Architectures/"
architecture_name = "MobileNet"
log_path = "./log/"
log_file = architecture_name + "_128x256.txt"
if os.path.isdir(log_path):
    print(log_path + " already exists !")
    if os.path.isfile(log_path + log_file):
        print("Rewriting " + log_file)
    else:
        print("Creating " + log_file)
        os.mknod(log_path + log_file)
else:
    print("Creating folder " + log_path)
    print("Creating " + log_file)
    os.mkdir(log_path)
    os.mknod(log_path + log_file)
sys.stdout = open(log_path + log_file, "w")

in_width, in_height, in_channel, out_width, out_height, out_channel, kernel_width, kernel_height, strides, pads, layer_name, layer_type = architecture_params(
    architecture_path + architecture_name + ".json")

axon_size = 128
neuron_size = 256

# Printing Parameters
print("*---------------------------------------------------------------------*")
print("Parameters extracted from the JSON architecture file")
print("Activation input width", in_width)
print("Activation input height", in_height)
print("Activation input maps", in_channel)
print("Activation output width", out_width)
print("Activation output height", out_height)
print("Activation output maps", out_channel)
print("Kernel widths", kernel_width)
print("Kernel heights", kernel_height)
print("Stride:", strides)
print("Paddings:", pads)
print("Layer name:", layer_name)
print("Layer type:", layer_type)
print("*---------------------------------------------------------------------*")

# Initialization of parameters for N_row_N_col_per_layer function  (Neuron_row and Neuron_col calculation for optimization mapping)
number_of_cores = np.array(np.zeros((len(out_width))), dtype=int)
neurons_per_core_utilization = np.array(np.zeros((len(out_width))), dtype=int)
synapse_per_core_utilization = np.array(np.zeros((len(out_width))), dtype=int)
N_row = np.array(np.zeros((len(out_width))), dtype=int)
N_col = np.array(np.zeros((len(out_width))), dtype=int)

print("Layer-wise details of mapping CNN architecture onto Neuromorphic Chip")

# loop for the entire layers in architecture
for l in range(len(out_width)):
    print("*-----------------------------------------------------------------*")
    print("layer %d => %d x %d x %d" % (l, out_width[l], out_height[l], out_channel[l]))
    number_of_cores[l], neurons_per_core_utilization[l], synapse_per_core_utilization[l], N_row[l], N_col[l] = core_utilization_per_layer(
        axon_size, neuron_size, in_width[l], in_height[l], in_channel[l], kernel_height[l], kernel_width[l], strides[l], out_channel[l], pads[l], layer_type[l])

    print("%d cores for layer %d:" % (number_of_cores[l], l))
    print("Maximum core utilization in layer %d: [" % (l), synapse_per_core_utilization[l], neurons_per_core_utilization[l], ']')
    print("Optimized size of neurons selected per core :[", N_row[l], ',', N_col[l], ']')

print('total cores', sum(number_of_cores))
