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


# Function 1 for efficient Batch-wise kron product
def kronecker_product1(a, b):
    # TODO(NOTE): What about autograd??

    # assert a.dim() == 3 and b.dim() == 3
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out


# Function 2 for efficient Batch-wise kron product, it is the fastest
def kronecker_product_einsum_batched(A: torch.Tensor, B: torch.Tensor):
    """
    Batched Version of Kronecker Products
    :param A: has shape (b, a, c)
    :param B: has shape (b, k, p)
    :return: (b, ak, cp)
    """
    # TODO(NOTE): What about autograd?? (we do not need to care about it here)

    assert A.dim() == 3 and B.dim() == 3
    res = torch.einsum('bac,bkp->bakcp', A, B).view(A.size(0),
                                                    A.size(1) * B.size(1),
                                                    A.size(2) * B.size(2)
                                                    )
    return res


class Encoding_Circuit(tq.QuantumModule):
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

        self.pre_gate = args.pre_gate  # string
        if args.ry_angle_factor_list is not None:
            self.theta_list = [theta * math.pi for theta in args.ry_angle_factor_list]
        else:
            self.theta_list = None

        self.entag_pattern = args.entag_pattern  # string
        self.permutation_list = args.permutation_list   # It could be None

        # quantum related
        self.q_device = tq.QuantumDevice(n_wires=self.num_tol_qubits)
        self.device = args.device   # run on cuda/cpu
        # self.measure = tq.MeasureAll(tq.PauliZ) # Do not need it

    def _data_add_qubits(self, x):  # It will be called by forward first, handle a batch of data
        bsz = x.shape[0]    # The shape of inputs is torch.Size([BS, #C, #W, #H])
        original_states = x.view(bsz, -1, 1).to(C_DTYPE)  # TODO(Note): Change to the complex type
        original_states = original_states.to(self.device)  # [BS, # Amplitude, 1]

        single_added_states = torch.zeros(2 ** self.num_enc_qubits, dtype=C_DTYPE).to(self.device)   # [#Added_Amplitude]
        single_added_states[0] = 1 + 0j  # to make it the zero state
        repeat_times = [bsz] + [1] * len(single_added_states.shape)  # repeat for batch size -> [BS, #Added_Amplitude]
        batch_added_states = single_added_states.repeat(*repeat_times).to(self.device)
        batch_added_states = batch_added_states.view(bsz, -1, 1).to(self.device)    # [BS, #Added_Amplitude, 1]

        x = kronecker_product_einsum_batched(original_states, batch_added_states)   # [BS, ..., 1]
        x = torch.squeeze(x, dim=-1)    # [BS, ...]
        reshape_dim = [bsz] + [2] * self.num_tol_qubits
        x = torch.reshape(x, reshape_dim).to(self.device)

        return x    # The output is (BS, 2, 2, ...)

    # def set_angle_list(self, angle_list):
    #     self.theta_list = [theta * math.pi for theta in angle_list]

    @tq.static_support
    def forward(self, x):
        # Add qubits to the batch of input data first
        x = self._data_add_qubits(x)    # (BS, 2, 2, ...)

        # Encode the data to self.states directly (add set method)
        self.q_device.set_states(x)

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

        # Return the encoded data
        x = self.q_device.get_states_1d().to(self.device)  # [BS, 2 ** num_tol_qubits]
        # print("The dtype inside encoding circuit is ", x.dtype)
        return x


def prep_enc_circuit_generation(args):
    # TODO(NOTE): Set the necessary parameters in args correctly

    num_ori_qubits = args.num_ori_qubits
    num_enc_qubits = args.num_enc_qubits
    total_qubits = num_ori_qubits + num_enc_qubits

    pre_gate = args.pre_gate
    Is_random_angles = args.Is_random_angles
    # ry_angle_factor_list = None  # if pre_gate is not ry

    entag_pattern = args.entag_pattern
    fh_num_cnot = args.fh_num_cnot
    lh_num_cnot = args.lh_num_cnot
    permutation_list = []   # 2-dim, [[c, t], [c, t], ...], ordered

    # TODO(NOTE): Preparation for building encryption circuit
    if pre_gate == 'ry' and Is_random_angles:
        # Generate the random angles for ry
        n_enc_qubits = args.num_enc_qubits
        ry_random_lb = args.random_angle_lb
        ry_random_ub = args.random_angle_ub
        ry_angle_factor_list = np.random.uniform(ry_random_lb, ry_random_ub, (n_enc_qubits)).tolist()
        args.ry_angle_factor_list = ry_angle_factor_list

    if entag_pattern == 'random':
        # Generate the random permutation
        for i in range(fh_num_cnot):  # for the fist half
            cnot_wires = np.random.choice(a=num_ori_qubits, size=2, replace=False, p=None).tolist()
            permutation_list.append(cnot_wires)

        for i in range(lh_num_cnot):  # for the last half
            c_idx = np.random.choice(a=np.arange(num_ori_qubits, total_qubits), replace=False, p=None)
            t_idx = np.random.choice(a=num_ori_qubits, replace=False, p=None)
            cnot_wires = [c_idx, t_idx]
            permutation_list.append(cnot_wires)

        args.permutation_list = permutation_list
    else:
        args.permutation_list = None


# def enc_circuit_generation(args):
#     prep_enc_circuit_generation(args)
#     # Build encryption circuit in pytorch
#     enc_circuit = Encoding_Circuit(args)
#
#     return enc_circuit

