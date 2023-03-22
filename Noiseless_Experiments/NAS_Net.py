import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.nn as nn

import torchquantum as tq
import torchquantum.functional as tqf

from NAS_dict import layer_name_dict, qiskit_layer_name_dict
from typing import Union, List
import numpy as np
from torchquantum.macro import C_DTYPE


def Default_Measure(q_device: tq.QuantumDevice, wires: Union[int, List[int]], device: Union['cuda', 'cpu']):

    """
    param: wires -> the index list of qubits, e.g., 5 quibts -> [0, 1, .., 4]
    This function is modified from expval in measure.py, used by Module MeasureAll(tq.QuantumModule)
    This function get the probability of |1> for each single qubit!
    """

    all_dims = np.arange(q_device.states.dim())  # e.g., torch.Size([3, 8]) -> dim = 2, equal to len of wires + 1 (bs)
    if isinstance(wires, int):
        wires = [wires]

    states = q_device.states    # [BS, 2, ..., ...]
    # compute magnitude
    state_mag = torch.abs(states) ** 2  # Get the probability of all the states, []

    expectations = []
    for wire in wires:
        # compute marginal magnitude
        reduction_dims = np.delete(all_dims, [0, wire + 1])  # index 0 for bs dim
        # get the probability for each single qubit (includes |0> and |1>)
        probs = state_mag.sum(list(reduction_dims)).to(C_DTYPE)     # [BS, 2]
        # print("The dtype inside Measure before is ", probs.dtype)
        # print("The device inside Measure before is ", probs.device)
        res = torch.index_select(probs, dim=1, index=torch.tensor([1]).to(device))  # [BS, 1], prob of |1> for each qubit
        res = res.squeeze(-1)   # [BS, 1] -> [BS]
        expectations.append(res)
    x = torch.stack(expectations, dim=-1)   # [BS, # selected quibts]
    # print("The dtype inside Measure before return is ", x.dtype)
    # print("The device inside Measure before is ", x.device)
    return x


class VQC_Net(tq.QuantumModule):
    """
    Define the general form of VQC Net.
    param: layer_list is the list of layer names (str), which could be used to find the layer class.
           layer_list should be in sequential order
           layer_list is the DNA sequence (hyperparameters)

           n_output is number of classes.

    NOTE: The input to forward is data with shape (bs, ...), do not need to be C_DTYPE
    NOTE: The output is (BS, n_output) with float type, we use probability instead of amplitude
    NOTE: The interface of different layers should be same/consistent
    """

    def __init__(self, layer_list, args):
        super().__init__()
        self.n_qubits = args.num_ori_qubits + args.num_enc_qubits  # total qubits
        self.n_output = args.output_num  # number of classes
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)  # It will be passed to the layer
        self.device = args.device   # GPU/CPU
        self.states_shape = [2] * self.n_qubits

        self.q_layers = nn.ModuleList()
        self.layer_list = []
        for name in layer_list:
            layer = layer_name_dict[name]  # Get the layer class w/o initialize
            if layer is not None:   # at least has one layer
                self.q_layers.append(layer(self.n_qubits))
                self.layer_list.append(name)

        # self.q_layer = self.QLayer()
        # self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        """
        Assume that the input x is the original input data (bs, c, w, h)
        But it could also handle input like (BS, 2**num_tol_qubits)
        """
        # encode data to the quantum device
        bsz = x.shape[0]
        reshape_dim = [bsz] + self.states_shape
        states = torch.reshape(x, reshape_dim).to(C_DTYPE)
        states = states.to(self.device)

        # # TODO(Note): data should be in Ctype!
        # print("The dtype inside VQC is ", states.dtype)
        # print("The shape of states in the model is ", states.shape)
        self.q_device.set_states(states)   # set the states to device, Now the data has been encoded to the device

        # states = self.q_device.states
        # print("The shape of states in the model before loop is ", states.shape)
        # start computation of quantum gates
        for layer in self.q_layers:
            layer(self.q_device)
            # states = self.q_device.states
            # print("The shape of states in the model within loop is ", states.shape)

        # post-process to get the meaningful results
        # TODO (NOTE): What output do we want to handle
        if self.n_output <= self.n_qubits:
            # use the first n_output qubits for classifying
            # states = self.q_device.states
            # print("The shape of states in the model is ", states.shape)
            x = Default_Measure(self.q_device, list(range(self.n_output)), self.device)  # [BS, n_output]
            # print("The shape of output in the model is ", x.shape)
        elif self.n_output <= 2**self.n_qubits:
            # use the first n_output probability for classifying
            x = self.q_device.states  # (BS, 2, 2, ...)
            # print("The dtype in another output case is ", x.dtype)   # C_type
            # print("The device in another output case is ", x.device)    # CUDA
            x = torch.abs(x) ** 2   # probability
            x = torch.reshape(x, [bsz, 2**self.n_qubits])
            x = torch.index_select(x, dim=1, index=torch.tensor(list(range(self.n_output))).to(self.device))    # (BS, n_output)
            # print("The dtype in another output case after is ", x.dtype)   # torch.float32, but does not matter
            # print("The device in another output case after is ", x.device)    # CUDA
        else:
            raise Exception("The number of classes is larger than the number of amplitudes")

        # x = F.log_softmax(x, dim=1)
        # TODO(Note): Transfer to float
        x = x.float()
        return x

    def get_layer_para_list(self):
        layer_para_list = []

        for layer in self.q_layers:
            para_dict = layer.get_para_dict()
            layer_para_list.append(para_dict)

        return layer_para_list

    def get_layer_list(self):
        return self.layer_list
