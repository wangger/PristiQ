import torch
import torch.nn as nn
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import sys
from utils import *
import math
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F

import torchquantum.functional as tqf
from torchquantum.macro import C_DTYPE


class Encoding_Circuit_Qiskit(tq.QuantumModule):
    """
    forward:
        Input: original image x from dataloader. (bs, c, w, h)
        Output: encoded state vectors. (BS, 2**num_tol_qubits)
    """
    def __init__(self, args):
        super().__init__()
        self.num_ori_qubits = args.num_ori_qubits
        self.num_enc_qubits = args.num_enc_qubits
        self.num_tol_qubits = self.num_ori_qubits + self.num_enc_qubits
        self.start_loc = self.num_ori_qubits  # start index of added qubit

        # encryption settings
        self.pre_gate = args.pre_gate  # string
        if args.ry_angle_factor_list is not None:
            self.theta_list = [theta * math.pi for theta in args.ry_angle_factor_list]
        else:
            self.theta_list = None

        self.entag_pattern = args.entag_pattern  # string
        self.permutation_list = args.permutation_list   # It could be None

        # quantum related
        # self.q_device = tq.QuantumDevice(n_wires=self.num_tol_qubits)
        self.device = args.device   # run on cuda/cpu
        # self.measure = tq.MeasureAll(tq.PauliZ) # Do not need it

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device

        # Add some more non-parameterized gates (add on-the-fly) in wiki order!!!!
        # copy + scale
        if self.pre_gate == 'ry':
            for offset, theta in enumerate(self.theta_list):    # self.theta_list cannot be None
                tqf.ry(self.q_device, wires=self.start_loc + offset, params=[theta], static=self.static_mode,
                       parent_graph=self.graph)
        elif self.pre_gate == 'hadamard':
            for index in range(self.start_loc, self.num_tol_qubits):
                tqf.hadamard(self.q_device, wires=index, static=self.static_mode, parent_graph=self.graph)
        elif self.pre_gate == 'identity':
            # print("Pre_gate in id branch")
            pass  # No scaling operation for the copied input
        else:
            raise Exception("The gate is not supported by the encoding circuit!")

        # permutation part
        if self.entag_pattern == 'random':
            for cnot_wires in self.permutation_list:  # cannot be None
                tqf.cnot(self.q_device, wires=cnot_wires, static=self.static_mode, parent_graph=self.graph)

        elif self.entag_pattern == 'single':
            # TODO (NOTE): The index setting is critical
            offset = self.num_ori_qubits    # used for target qubit loc
            for index in range(self.start_loc, self.num_tol_qubits):
                # This is for a single CNOT
                index_c = index  # loc of the added qubit
                index_r = index_c - offset  # The original qubit is controlled by the added qubit
                tqf.cnot(self.q_device, wires=[index_c, index_r], static=self.static_mode, parent_graph=self.graph)
                # # inverse order
                # tqf.cnot(self.q_device, wires=[index_r, index_c], static=self.static_mode, parent_graph=self.graph)

        elif self.entag_pattern == 'full':
            for c_index in range(self.start_loc, self.num_tol_qubits):
                index_c = c_index  # loc of the added qubit
                for t_index in range(self.num_ori_qubits):  # entangle each added qubit with all of the original qubits
                    index_r = t_index  # Each original qubit is controlled by the added qubit
                    tqf.cnot(self.q_device, wires=[index_c, index_r], static=self.static_mode, parent_graph=self.graph)
                    # # inverse order
                    # tqf.cnot(self.q_device, wires=[index_r, index_c], static=self.static_mode,
                    # parent_graph=self.graph)

        elif self.entag_pattern == 'identity':
            # print("Entanglement in id branch")
            pass  # No permutation operation for the copied input

        elif self.entag_pattern == 'single_add_0':
            for i in range(0, self.num_ori_qubits, 2):  # the depth is only 1
                if i > self.num_ori_qubits - 2:
                    break
                else:
                    tqf.cnot(self.q_device, wires=[i, i+1], static=self.static_mode, parent_graph=self.graph)
            # TODO (NOTE): The index setting is critical
            offset = self.num_ori_qubits    # used for target qubit loc
            for index in range(self.start_loc, self.num_tol_qubits):
                # This is for a single CNOT
                index_c = index  # loc of the added qubit
                index_r = index_c - offset  # The original qubit is controlled by the added qubit
                tqf.cnot(self.q_device, wires=[index_c, index_r], static=self.static_mode, parent_graph=self.graph)

        elif self.entag_pattern == 'single_add_1':
            for i in range(0, self.num_ori_qubits-1):  # the depth is only 1
                tqf.cnot(self.q_device, wires=[i, i + 1], static=self.static_mode, parent_graph=self.graph)
            # TODO (NOTE): The index setting is critical
            offset = self.num_ori_qubits  # used for target qubit loc
            for index in range(self.start_loc, self.num_tol_qubits):
                # This is for a single CNOT
                index_c = index  # loc of the added qubit
                index_r = index_c - offset  # The original qubit is controlled by the added qubit
                tqf.cnot(self.q_device, wires=[index_c, index_r], static=self.static_mode, parent_graph=self.graph)
        else:
            raise Exception("This type of permutation is not currently supported!")