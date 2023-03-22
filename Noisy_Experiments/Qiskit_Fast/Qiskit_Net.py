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
from torchquantum.plugins import tq2qiskit
from qiskit import QuantumCircuit


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


def qiskit_measurement(q_device: tq.QuantumDevice):
    circ = QuantumCircuit(q_device.n_wires, q_device.n_wires)
    circ.measure(list(range(q_device.n_wires)), list(range(q_device.n_wires)))
    return circ


def amp_qiskit_compose_circs(list_data_circ, comp_circ, measurement_circ, q_device: tq.QuantumDevice):
    circs_all = []
    reverse_mapping = list(range(q_device.n_wires - 1, -1, -1))  # reverse mapping
    for data_cir in list_data_circ:
        # print(data_cir.num_clbits)
        # print(data_cir.num_qubits)
        # print(comp_circ.num_clbits)
        # print(comp_circ.num_qubits)
        data_cir.compose(comp_circ, qubits=reverse_mapping, inplace=True)
        data_cir.compose(measurement_circ, inplace=True)
        circs = data_cir
        circs_all.append(circs)

    return circs_all


def tq2qiskit_expand_params_amp(x: torch.Tensor, num_ori_qubits, num_enc_qubits):
    """Expand the input classical values to amplitude. No
       Qiskit.circuit.Parameter is used. All the amplitude are hard encoded.
       This will solve the issue of qiskit bugs.

       [NOTE]: It could handle multiple security encoding

        Args:
            q_device (tq.QuantumDevice): Quantum device
            x (torch.Tensor): Input classical values waited to be embedded in the
                circuits.

        Returns:
            circ_all (List[Qiskit.QiskitCircuit]): expand the parameters into encodings
                and return the hard coded circuits.
    """
    circ_all = []
    n_qubits = num_ori_qubits + num_enc_qubits
    feature_loc = list(range(num_enc_qubits, n_qubits))

    for sample in x:
        sample = sample.reshape(-1)  # (2**num_tol_qubits)
        sample = sample.numpy()

        import transformations as trafo
        sample = trafo.unit_vector(sample)
        sample_qc = QuantumCircuit(n_qubits, n_qubits)
        sample_qc.initialize(sample, feature_loc)

        circ_all.append(sample_qc)

    return circ_all


class Qiskit_Net(tq.QuantumModule):
    """
    Define the general form of the Encoding circuit (if necessary) + VQC Net.


    param: layer_list is the list of layer names (str), which could be used to find the layer class.
           layer_list should be in sequential order
           layer_list is the DNA sequence (hyperparameters)

           enc_cir_qiskit: Encoding_Circuit_Qiskit(tq.QuantumModule) from Qiskit_Fast.S_Encode

    NOTE: The input to forward is data with shape (bs, ...), do not need to be C_DTYPE
    NOTE: The output is (BS, n_output) with float type, we use probability instead of amplitude
    NOTE: The interface of different layers should be same/consistent
    """

    def __init__(self, layer_list, args, enc_circ_qiskit=None):
        super().__init__()
        self.n_output = args.output_num  # number of classes
        self.device = args.device   # GPU/CPU
        self.Is_S_Encoding = args.Is_S_Encoding
        # if self.Is_S_Encoding:
        #     self.n_qubits = args.num_ori_qubits + args.num_enc_qubits  # total qubits
        # else:
        #     self.n_qubits = args.num_ori_qubits

        self.n_ori_qubits = args.num_ori_qubits
        self.n_enc_qubits = args.num_enc_qubits
        self.n_qubits = args.num_ori_qubits + args.num_enc_qubits  # total qubits, enc_qubits=0 if for baseline
        self.states_shape = [2] * self.n_qubits

        # quantum circuit related
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)  # It will be passed to the layer
        self.enc_circ = enc_circ_qiskit  # qiskit circuit # tq.QuantumModule for encryption

        # TODO(NOTE): Build the torch module for computation
        self.q_layers = nn.ModuleList()
        # self.layer_list = []
        for name in layer_list:
            layer = layer_name_dict[name]  # Get the layer class w/o initialize
            if layer is not None:   # at least has one layer
                self.q_layers.append(layer(self.n_qubits))
                # self.layer_list.append(name)

        # self.q_layer = self.QLayer()
        # self.measure = tq.MeasureAll(tq.PauliZ)

        self.q_layers_circ = None
        self.Not_Expanded = args.Not_Expanded

    def set_layer_circ(self, layer_circ):
        self.q_layers_circ = layer_circ

    def forward(self, x, use_qiskit=True):
        """
        Assume that the input x is the original input data (bs, c, w, h)
        But it could also handle input like (BS, 2**num_tol_qubits), only the BS dim is important

        [NOTE]: For encryption pattern, the x should the one after operation of increasing qubits. This operation
        is done outside
        """
        if use_qiskit:
            # a list of circuit for amplitude encoding, [bs, ]
            list_data_circ = tq2qiskit_expand_params_amp(x, self.n_ori_qubits, self.n_enc_qubits)

            # Build the quantum circuit for q layer
            # q_layers_circ = QuantumCircuit(self.q_device.n_wires)
            # for layer in self.q_layers:
            #     print(type(layer))
            #     print(layer)
            #     layer_circ = tq2qiskit(self.q_device, layer)
            #     q_layers_circ.compose(layer_circ, inplace=True)

            q_layers_circ = self.q_layers_circ

            if self.Not_Expanded:
                n_qubits = self.n_ori_qubits
            else:
                n_qubits = self.n_qubits
            mapping = list(range(n_qubits))

            if self.Is_S_Encoding:  # self.enc_circ is not None
                # comp_circ = tq2qiskit(self.q_device, self.enc_circ)
                comp_circ = self.enc_circ
                # comp_circ.compose(q_layers_circ, qubits=mapping, inplace=True)
                comp_circ = comp_circ.compose(q_layers_circ, qubits=mapping)
            else:
                comp_circ = QuantumCircuit(self.n_qubits, self.n_qubits)
                comp_circ.compose(q_layers_circ, qubits=mapping, inplace=True)

            measurement_circ = qiskit_measurement(self.q_device)

            # TODO(NOTE): It includes a reverse connection between data_circ and comp_circ
            assembled_circs = amp_qiskit_compose_circs(list_data_circ, comp_circ, measurement_circ, self.q_device)

            # The output has already been process to the one you want
            x0 = self.qiskit_processor.amp_process_ready_circs(self.q_device, assembled_circs,
                                                               self.n_output).to(self.device)    # torch.Tensor
            x = x0

        else:
            # encode data to the quantum device
            bsz = x.shape[0]
            reshape_dim = [bsz] + self.states_shape

            # TODO: If use_qiskit = False and is_encoding, expand x is necessary

            # print(x.shape)
            # print(reshape_dim)
            states = torch.reshape(x, reshape_dim).to(C_DTYPE)
            states = states.to(self.device)

            # # TODO(Note): data should be in Ctype!
            # print("The dtype inside VQC is ", states.dtype)
            # print("The shape of states in the model is ", states.shape)
            self.q_device.set_states_me(states)   # set the states to device, Now the data has been encoded to the device

            # TODO(NOTE): Pass the input data to the enc circuit
            if self.Is_S_Encoding:  # self.enc_circ is not None
                self.enc_cir(self.q_device)

            # states = self.q_device.states
            # print("The shape of states in the model before loop is ", states.shape)
            # start computation of quantum gates

            # calculate using the VQC Net
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

        # TODO(Note): Transfer to float
        x = x.float()
        # x = F.log_softmax(x, dim=1)
        # x = torch.log(x + 1e-8)
        return x


def Testing_Ideal_Torch(model, test_loader, args, enc_circuit=None):  # test ideally
    model.eval()
    correct = 0
    iter = 0
    total_data = 0  # TODO(NOte): We use drop last for data loader
    Is_S_Encoding = args.Is_S_Encoding
    device = args.device

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # TODO(Note): encoding the data before computing
            if Is_S_Encoding:
                data = enc_circuit(data)    # [BS, 2 ** num_tol_qubits]

            output = model(data)  # output value is prob
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # print(pred.shape)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            iter += 1
            total_data += len(data)

    final_acc = 100. * float(correct) / float(total_data)
    print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(correct, len(test_loader.dataset), final_acc))
    return final_acc / 100


def Testing_Noisy_Sim_Qiskit(qiskit_model, test_loader):  # test on noisy quantum simulator
    """
    return: acc < 1
    """
    # test_loss = 0
    correct = 0
    iter = 0
    total_data = 0  # TODO(NOte): We use drop last for data loader
    # Is_S_Encoding = args.Is_S_Encoding

    for batch_idx, (data, target) in enumerate(test_loader):
        output = qiskit_model(data, use_qiskit=True)  # output value is prob
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        iter += 1
        total_data += len(data)

    final_acc = 100. * float(correct) / float(total_data)
    print('Test set: Average Accuracy: {}/{} ({:.2f}%)'.format(correct, len(test_loader.dataset), final_acc))
    return final_acc / 100


def Testing_QC_Qiskit(qiskit_model, test_loader):  # test on noisy quantum simulator
    """
    return: acc < 1
    """
    # test_loss = 0
    correct = 0
    iter = 0
    total_data = 0  # TODO(NOte): We use drop last for data loader
    # Is_S_Encoding = args.Is_S_Encoding

    for batch_idx, (data, target) in enumerate(test_loader):
        output = qiskit_model(data, use_qiskit=True)  # output value is prob
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        iter += 1
        total_data += len(data)

    final_acc = 100. * float(correct) / float(total_data)
    print('Test set: Average Accuracy: {}/{} ({:.2f}%)'.format(correct, len(test_loader.dataset), final_acc))
    return final_acc / 100
