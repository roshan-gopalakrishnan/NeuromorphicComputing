# Coded by Roshan Gopalakrishnan, contact: roshan.gopalakrishnan@gmail.com

import math
import json
import numpy as np


def architecture_params(architecture_json_filename):
    # architecture Parameters from JSON architecture file

    with open(architecture_json_filename) as arch:
        data = json.loads(arch.read())

    arch = list(data.values())         # for python 3; for python 2 data.values()[0]

    layer_name = []
    layer_type = []
    input_acts = []
    output_acts = []
    single_act_size = []
    kernels = []
    single_kernel_size = []
    strides = []
    pads = []
    x_index = []
    input_acts_width = []
    input_acts_height = []
    input_acts_channel = []
    output_acts_width = []
    output_acts_height = []
    output_acts_channel = []
    # for future use
    kernel_x_index = 0
    kernel_width = []
    kernel_height = []

    for i in range(len(arch[0])):
        layer_name.append(str(arch[0][i]["Layer"][0]["name"]))
        layer_type.append(str(arch[0][i]["Layer"][0]["type"]))
        input_acts.append(str(arch[0][i]["details"][0]["activations"][0]["input"]))
        output_acts.append(str(arch[0][i]["details"][0]["activations"][0]["output"]))
        if len(input_acts[i]) < 5:
            print("Input error: check the input activation size in the JSON architecture file")
        else:
            for j in range(len(input_acts[i])):
                if input_acts[i][j] == 'x':
                    x_index.append(j)
            if len(x_index) == 2:
                input_acts_width.append(int(input_acts[i][0:x_index[0]]))
                input_acts_height.append(int(input_acts[i][x_index[0] + 1: x_index[1]]))
                input_acts_channel.append(int(input_acts[i][x_index[1] + 1:]))
            else:
                print("Input error: check the format of input activation size in the JSON architecture file")
            x_index = []
            if input_acts_width[i] == input_acts_height[i]:
                # print "Activation size is square!"
                single_act_size.append(int(input_acts_width[i]))
            else:
                single_act_size.append("$")
                # print "Activation size is rectangular rather than square!"
        if len(output_acts[i]) < 5:
            print("Output error: check the output activation size in the JSON architecture file")
        else:
            for j in range(len(output_acts[i])):
                if output_acts[i][j] == 'x':
                    x_index.append(j)
            if len(x_index) == 2:
                output_acts_width.append(int(output_acts[i][0:x_index[0]]))
                output_acts_height.append(int(output_acts[i][x_index[0] + 1: x_index[1]]))
                output_acts_channel.append(int(output_acts[i][x_index[1] + 1:]))
            else:
                print("Output error: check the format of output activation size in the JSON architecture file")
            x_index = []
            if output_acts_width[i] == output_acts_height[i]:
                # print "Activation size is square!"
                single_act_size.append(int(output_acts_width[i]))
            else:
                single_act_size.append("$")
                # print "Activation size is rectangular rather than square!"
        kernels.append(str(arch[0][i]["details"][0]["kernel"]))
        if len(kernels[i]) < 3:
            print("Input error: check the kernel size in the JSON architecture file")
        else:
            for j in range(len(kernels[i])):
                if kernels[i][j] == 'x':
                    kernel_x_index = j
            if kernel_x_index != 0:
                kernel_width.append(int(kernels[i][0:kernel_x_index]))
                kernel_height.append(int(kernels[i][kernel_x_index + 1:]))
            else:
                print("Input error: check the format of kernel size in the JSON architecture file")
            if kernel_width[i] == kernel_height[i]:
                # print "Kernel size is square!"
                single_kernel_size.append(int(kernel_width[i]))
            else:
                single_kernel_size.append("$")
                # print "Kernel size is rectangular rather than square!"
        strides.append(arch[0][i]["details"][0]["stride"])
        pads.append(arch[0][i]["details"][0]["padding"])

    return input_acts_width, input_acts_height, input_acts_channel, output_acts_width, output_acts_height, output_acts_channel, kernel_width, kernel_height, strides, pads,layer_name, layer_type


def backward_activations(width, kernel_width, strides):
    # Calculate the activations needed for previous layer in backward manner

    acts = np.ones(len(kernel_width))
    if (len(strides) == len(kernel_width) == len(width)):
        for i in range(1, len(kernel_width)):
            acts[-(i + 1)] = (acts[-i] - 1) * strides[-i] + kernel_width[-i]
            if (acts[-(i + 1)] > width[-(i + 1)]):
                acts[-(i + 1)] = width[-(i + 1)]
    return acts


def namingNeurons(layer_name, input_size, channels):
    # Name each neurons in every layers
    # Naming is done as example given below
    # L0-F0:N[1,1] => Layer 0, feature map 0, Neuron (1,1)

    LayerString = layer_name #'L' + str(layer_number)
    layerAfterNaming = []
    for inputChannel in range(1, channels + 1):
        NeuronsInEachFeatureMapsinaLayer_2dArray = []
        for k in range(1, input_size + 1):
            temp = []
            for l in range(1, input_size + 1):
                temp.append(LayerString + '- F%s :N[%s,%s]' % (inputChannel, k, l))
            NeuronsInEachFeatureMapsinaLayer_2dArray.append(np.array(temp))
        layerAfterNaming.append(NeuronsInEachFeatureMapsinaLayer_2dArray)

    return np.array(layerAfterNaming)


def axon_selection(axon_size, neuron_size, kernel_height, kernel_width, input_channel, output_width, output_height, stride, channel_batch_on_core):
    # find the factors of channel_batch_on_core
    # function called in row_column_per_layer
    factors = []
    for i in range(1, channel_batch_on_core + 1):
        if (channel_batch_on_core % i == 0):
            if (i<=output_height and channel_batch_on_core/i<=output_width):
                factors.append(i)
    # print 'factors', factors

    N_row = np.zeros(len(factors))
    N_col = np.zeros(len(factors))
    N_axon = np.zeros(len(factors))
    for i, j in enumerate(factors):
        N_row[i] = j
        N_col[i] = int(math.floor(channel_batch_on_core / j))
        N_axon[i] = int(kernel_height * kernel_width * input_channel + kernel_height * stride * input_channel * \
            (N_col[i] - 1) + kernel_width * stride * input_channel * (N_row[i] - 1) + \
            stride * stride * input_channel * (N_col[i] - 1) * (N_row[i] - 1))
        # print(i, 'factors:', j, '[Row, Col] [', N_row[i], N_col[i], '] N_axon:', N_axon[i])
        # print 'kernel_height',kernel_height,'stride',stride

    return N_row, N_col, N_axon


def row_column_per_layer(axon_size, neuron_size, kernel_height, kernel_width, input_channel, output_width, output_height, output_channel, stride, toeplitz):
    # Row and column selection per layer for each core
    # function called in N_row_N_col_per_layer

    if toeplitz:
        output_channel = 1
        channel_batch_on_core = int(math.floor(neuron_size / output_channel))
        if (channel_batch_on_core > output_width * output_height):
            channel_batch_on_core = int(output_width * output_height)
    else:
        channel_batch_on_core = int(math.floor(neuron_size / output_channel))
    print('First calculation of channel batch on core', channel_batch_on_core)
    N_row, N_col, N_axon = axon_selection(axon_size, neuron_size, kernel_height, kernel_width,
                                          input_channel, output_width, output_height, stride, channel_batch_on_core)
    # If all N_axon is out of axon size
    if (all(N_axon > axon_size)):
        for i in range(0, channel_batch_on_core, 2):
            # print "Axon selection: recalculating row and column"
            if (channel_batch_on_core == 2 or channel_batch_on_core % 2 != 0):
                channel_batch_on_core -= 1
            else:
                channel_batch_on_core -= 2
            print("channel_batch_on_core", channel_batch_on_core)
            N_row, N_col, N_axon = axon_selection(axon_size, neuron_size, kernel_height, kernel_width,
                                                  input_channel, output_width, output_height, stride, channel_batch_on_core)
            # at least one N_axon is within axon size
            if (any(N_axon <= axon_size)):
                print('Break!')
                break
    #value, index = min((value, index) for (index, value) in enumerate(N_axon))

    #return N_row[index], N_col[index], value

    # Optimized square matrix selection of Nrow and Ncol
    min_N_axon = [[index, value] for (index, value) in enumerate(N_axon) if value == min(N_axon)]
    #print("Index, value of min(N_axon)", min_N_axon)
    #print("length of min_N_axon", len(min_N_axon))
    indices = []
    values = []
    for i in range(len(min_N_axon)):
        indices.append(min_N_axon[:][i][0])
        values.append(min_N_axon[:][i][1])
    #print("indices and values", indices, values)
    N_row = N_row[indices]
    N_col = N_col[indices]
    #print('After min(N_axon): [Row, Col] [', N_row, N_col, '] N_axon:', min_N_axon)
    if len(N_row)==1:
        print("core utilization is {} x {}".format(N_row[0],N_col[0]))
        N_row = N_row[0]
        N_col = N_col[0]
        N_axon = values[0]
    elif (N_row.all() == N_col.all() and len(N_row)==2):
        print("core utilization is either {} x {} or {} x {}".format(N_row[0],N_col[0],N_row[1],N_col[1]))
        print("we are choosing {} x {}".format(N_row[0],N_col[0]))
        N_row = N_row[0]
        N_col = N_col[0]
        N_axon = values[0]
    elif (N_row.all() == N_col.all() and len(N_row)>2):
        sum_row_col = N_row[indices] + N_col[indices]
        #print("Sum of row and col", sum_row_col)
        min_of_sum, min_index = min((min_value, min_index) for min_index, min_value in enumerate(sum_row_col))
        #print("Sum of row and col/ min of sum", sum_row_col, min_index, min_of_sum)
        N_row = N_row[min_index]
        N_col = N_col[min_index]
        N_axon = N_axon[min_index]
        print("core utilization is {} x {}".format(N_row,N_col))

    print('[Row, Col] [', N_row, N_col, '] N_axon:', N_axon)

    return N_row, N_col, N_axon


def core_utilization_per_layer(axon_size, neuron_size, input_width, input_height, input_channel, kernel_height, kernel_width, stride, output_channel, padding, layer_type):
    # Calculates the number of cores, axon, neuron utilizations

    output_width = int(np.floor((input_width - kernel_width + 2 * padding) / stride + 1))
    output_height = int(np.floor((input_height - kernel_height + 2 * padding) / stride + 1))
    print("Input layer:", input_width, "X", input_height, "X", input_channel)
    print("Calculated Output layer:", output_width, "X", output_height, "X", output_channel)
    # 2 methods of core mapping - Toeplitz or Hybrid
    toeplitz = False
    if layer_type == "pooling" or layer_type == "depthwise":
        input_channel = 1
        toeplitz = True
        print("Kernel size:", kernel_width, "X", kernel_height)
    else:
        print("Kernel size:", kernel_width, "X", kernel_height, "X", output_channel)
    print("Stride:", stride)
    print("Padding:", padding)
    print("Layer type:", layer_type)
    # Synapse connections per neuron
    # Number of ReRAM connections per column in a core
    synapse_per_neuron = kernel_width * kernel_height * input_channel
    synapse_per_core_utilization = synapse_per_neuron
    core_utilization = 1
    # different cases of core utilization
    if toeplitz:
        if (synapse_per_neuron > axon_size):
            print("Toeplitz: this condition never occurs, otherwise axon_size must be small")
            N_row = 1
            N_col = 1
            print('Warning: Matrix splitting is needed')
            core_utilization = math.floor(synapse_per_neuron / axon_size)
            synapse_per_core_utilization = axon_size
            if (synapse_per_neuron % axon_size != 0):
                core_utilization += 1

            neurons_per_core_utilization = 1
            core_utilization = core_utilization + 1  # 1 is for adder
            # Total number of core utilization
            number_of_cores = output_width * output_height * core_utilization
        else:
            N_row, N_col, synapse_per_core_utilization = row_column_per_layer(
                axon_size, neuron_size, kernel_height, kernel_width, input_channel, output_width, output_height, output_channel, stride, toeplitz)
            neurons_per_core_utilization = N_row * N_col
            # Above number of cores considers each core for output channels
            # But if mapping section is small we can map multiple feature maps onto a single core
            # especially for depthwise layers while pipeline Mapping
            # Output channels per core =>
            OCPerCore = math.floor(min(axon_size/synapse_per_core_utilization, neuron_size/neurons_per_core_utilization))
            synapse_per_core_utilization = synapse_per_core_utilization * OCPerCore
            neurons_per_core_utilization = neurons_per_core_utilization * OCPerCore
            number_of_cores = math.ceil(
                output_width / N_col) * math.ceil(output_height / N_row) * math.ceil(output_channel / OCPerCore)

            print("Toeplitz mapping of {} feature maps onto single core".format(OCPerCore))
            print("N_row", N_row)
            print("N_col", N_col)
            print("neurons_per_core_utilization", neurons_per_core_utilization)
            print("synapse_per_core_utilization", synapse_per_core_utilization)
            print("Number of cores", number_of_cores)
    else:
        if (synapse_per_neuron > axon_size):
            N_row = 1
            N_col = 1
            print('Warning: Matrix splitting is needed')
            core_utilization = math.floor(synapse_per_neuron / axon_size)
            synapse_per_core_utilization = axon_size
            if (synapse_per_neuron % axon_size != 0):
                core_utilization += 1
            if (output_channel > neuron_size):
                print("Condition 1: Synapse Per Neuron > Axon Size & Output Channel > Neuron Size")
                core_utilization_col = math.floor(output_channel / neuron_size)
                neurons_per_core_utilization = neuron_size
                if (output_channel % neuron_size != 0):
                    core_utilization_col += 1
                core_utilization = core_utilization * core_utilization_col + \
                    core_utilization_col  # core_utilization_col for adders
                # Total number of core utilization
                #number_of_cores = output_width * output_height * core_utilization
            else:
                print("Condition 2: Synapse Per Neuron > Axon Size & Output Channel < Neuron Size")
                neurons_per_core_utilization = output_channel
                core_utilization = core_utilization + 1  # 1 is for adder
                # Total number of core utilization
            number_of_cores = output_width * output_height * core_utilization
        else:
            #synapse_per_core_utilization = synapse_per_neuron
            # Total convolutions in a convolution layer
            # Or total number of neurons needed in the single convolution layer

            if (output_channel > neuron_size):
                print("Condition 3: Synapse Per Neuron < Axon Size & Output Channel > Neuron Size")
                N_row = 1
                N_col = 1
                total_neurons_per_channel = output_width * output_height
                output_channel_batch = int(math.ceil(output_channel / neuron_size))
                number_of_cores = total_neurons_per_channel * output_channel_batch
                neurons_per_core_utilization = neuron_size
                print("Two different utilizations for cores")
            else:
                print("Condition 4: Synapse Per Neuron < Axon Size & Output Channel < Neuron Size")
                N_row, N_col, synapse_per_core_utilization = row_column_per_layer(
                    axon_size, neuron_size, kernel_height, kernel_width, input_channel, output_width, output_height, output_channel, stride, toeplitz)
                neurons_per_core_utilization = output_channel * N_row * N_col
                number_of_cores = math.ceil(output_width / N_col) * math.ceil(output_height / N_row)
                print("N_row", N_row)
                print("N_col", N_col)
                print("neurons_per_core_utilization", neurons_per_core_utilization)
                print("synapse_per_core_utilization", synapse_per_core_utilization)
                print("Number of cores", number_of_cores)
    if neurons_per_core_utilization > neuron_size:
        raise Exception('Neurons utilized per core {} should not exceed core neuron size {}.'.format(neurons_per_core_utilization, neuron_size))
    if synapse_per_core_utilization > axon_size:
        raise Exception('Axons utilized per core {} should not exceed core axon size {}.'.format(synapse_per_core_utilization, axon_size))
    if N_row > output_height:
        raise Exception('Neurons selected across row {} should not exceed output height {}.'.format(N_row, output_height))
    if N_col > output_width:
        raise Exception('Neurons selected across column {} should not exceed output width {}.'.format(N_col, output_width))

    return number_of_cores, neurons_per_core_utilization, synapse_per_core_utilization, N_row, N_col


def axon_neuron_weight_per_layer(inputlayer, input_afterPadding, kernel_width, kernel_height, strides, output_channel, N_row, N_col, neuron_size, axon_size, weight):
    # Outputs a list of axons and neurons along with weight for each cores
    # Weight format is axon size x neuron size

    axons_per_layer = []
    neurons_per_layer = []
    weights_per_layer = []
    core = 1
    # output_channel = 1
    # selection of N_rowxN_col window within a layer for each cores
    # Layer wise implementation
    for i in range(0, np.shape(inputlayer)[1], N_row):
        for j in range(0, np.shape(inputlayer)[2], N_col):
            # variables for a single core
            neurons = np.array(np.zeros(neuron_size), dtype=object)
            axons = np.array(np.zeros(axon_size), dtype=object)
            weight_matrix = np.array(np.zeros((axon_size, neuron_size)), dtype=object)
            axon_dummy = []

            # selection of N_rowxN_col neurons within each core
            # core wise implementation
            neuron_count = 0
            for m in range(0, N_row):
                for n in range(0, N_col):
                    # selecting those same axons for every neurons in the block of neuron
                    axons_4_neuron_block = input_afterPadding[:, (i + m) * strides:(i + m) * strides +
                                                              kernel_height, (j + n) * strides:(j + n) * strides + kernel_width]
                    # print "axons for neuron block with padding:", axons_4_neuron_block
                    # to remove padding
                    axons_4_neuron_block_wo_pad = [
                        c for a in axons_4_neuron_block for b in a for c in b if c != '1.0']
                    # print "axons for neuron block:", axons_4_neuron_block_wo_pad
                    # to remove overlapping axons for each neurons in a core
                    for item in axons_4_neuron_block_wo_pad:
                        if item not in axon_dummy:
                            axon_dummy.append(item)
                    # condition to remove the outlier incomplete convolutions
                    if (i + m < np.shape(inputlayer)[1] and j + n < np.shape(inputlayer)[2]):
                        # neuron block represents neurons across feature maps
                        # A single neuron from neuron block is selected here
                        for o in range(0, output_channel):
                            weight_index = []
                            neurons[neuron_count] = inputlayer[o, i + m, j + n]
                            # print "Neuron", neurons[neuron_count]
                            weight_index = [[ai, bi, ci] for ai, a in enumerate(
                                axons_4_neuron_block) for bi, b in enumerate(a) for ci, c in enumerate(b) if c != '1.0']
                            # print "Weights index for neuron block:", weight_index
                            # print len(weight_index), len(axons_4_neuron_block), len(axons_4_neuron_block_wo_pad)
                            if (len(weight_index) <= axon_size):
                                for l in range(0, len(weight_index)):
                                    # print l
                                    weight_matrix[axon_dummy.index(
                                        axons_4_neuron_block_wo_pad[l]), neuron_count] = weight[o, weight_index[l][0], weight_index[l][1], weight_index[l][2]]
                            else:
                                print("Warning: Core matrix splitting")
                                for l in range(0, axon_size):
                                    # print l
                                    weight_matrix[axon_dummy.index(
                                        axons_4_neuron_block_wo_pad[l]), neuron_count] = weight[o, weight_index[l][0], weight_index[l][1], weight_index[l][2]]
                                for l in range(0, len(weight_index) - axon_size):
                                    # print l
                                    # print axon_dummy.index(axons_4_neuron_block_wo_pad[l]), neuron_count
                                    weight_matrix[axon_dummy.index(axons_4_neuron_block_wo_pad[l]), neuron_count] = weight[o,
                                                                                                                           weight_index[axon_size + l][0], weight_index[axon_size + l][1], weight_index[axon_size + l][2]]
                            neuron_count += 1
            # print 'Axon dummy', axon_dummy, len(axon_dummy)
            if (len(axon_dummy) <= axon_size):
                axons[0:len(axon_dummy)] = axon_dummy
            else:
                print("Warning: Core matrix splitting")
                axons[0:axon_size] = axon_dummy[0:axon_size]
            axons_per_layer.append(axons)
            # print "Neuron", neurons
            neurons_per_layer.append(neurons)
            weights_per_layer.append(weight_matrix)
            # print "Weight matrix", weight_matrix[:, 0]
            print("core %d utilization: = [%d %d]" % (core, len(axon_dummy), neuron_count))
            core += 1

    return axons_per_layer, neurons_per_layer, weights_per_layer
