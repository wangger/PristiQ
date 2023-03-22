import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.nn as nn

import torchquantum as tq
import torchquantum.functional as tqf

# TODO(Note): The interface of layer should be consistent (in the same form)!


# Zhirui's Model (ry + circuit 10)
class QLayer_10(tq.QuantumModule):
    """
    The input to the forward is q_device directly (has the encoded states already)
    The output to the forward is still q_device (has the states after gate)
    """

    def __init__(self, n_qubits):
        super().__init__()
        # gates with trainable parameters
        self.n_qubits = n_qubits
        self.trainable_modules_head = nn.ModuleList()
        for i in range(self.n_qubits):
            self.trainable_modules_head.append(tq.RY(has_params=True, trainable=True))

        self.trainable_modules_tail = nn.ModuleList()
        for i in range(self.n_qubits):
            self.trainable_modules_tail.append(tq.RY(has_params=True, trainable=True))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        """
        1. To convert tq QuantumModule to qiskit or run in the static
        model, need to:
            (1) add @tq.static_support before the forward
            (2) make sure to add
                static=self.static_mode and
                parent_graph=self.graph
                to all the tqf functions, such as tqf.hadamard below
        """
        self.q_device = q_device  # no need to reset here, but get the new states for each batch

        # For the trainable head
        for index in range(self.n_qubits):
            self.trainable_modules_head[index](self.q_device, wires=index)

        # For the non-trainable entanglement (add on-the-fly)
        for offset in range(self.n_qubits - 1):
            loc_0 = self.n_qubits - 1 - offset
            loc_1 = loc_0 - 1
            tqf.cz(self.q_device, wires=[loc_0, loc_1], static=self.static_mode, parent_graph=self.graph)

        tqf.cz(self.q_device, wires=[0, self.n_qubits - 1], static=self.static_mode, parent_graph=self.graph)

        # For the trainable tail
        for index in range(self.n_qubits):
            self.trainable_modules_tail[index](self.q_device, wires=index)

    def get_para_dict(self):
        para_dict = {}
        para_dict['head'] = []
        for ry_module in self.trainable_modules_head:
            for name, parameters in ry_module.named_parameters():   # only one
                para_dict['head'].append(parameters.data.item())

        para_dict['tail'] = []
        for ry_module in self.trainable_modules_tail:
            for name, parameters in ry_module.named_parameters():   # only one
                para_dict['tail'].append(parameters.data.item())

        return para_dict


class QLayer_2(tq.QuantumModule):
    """
    The input to the forward is q_device directly (has the encoded states already)
    The output to the forward is still q_device (has the states after gate)
    """

    def __init__(self, n_qubits):
        super().__init__()
        # gates with trainable parameters
        self.n_qubits = n_qubits
        self.trainable_modules_head_rx = nn.ModuleList()
        for i in range(self.n_qubits):
            self.trainable_modules_head_rx.append(tq.RX(has_params=True, trainable=True))

        self.trainable_modules_head_rz = nn.ModuleList()
        for i in range(self.n_qubits):
            self.trainable_modules_head_rz.append(tq.RZ(has_params=True, trainable=True))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        """
        1. To convert tq QuantumModule to qiskit or run in the static
        model, need to:
            (1) add @tq.static_support before the forward
            (2) make sure to add
                static=self.static_mode and
                parent_graph=self.graph
                to all the tqf functions, such as tqf.hadamard below
        """
        self.q_device = q_device  # no need to reset here, but get the new states for each batch

        # For the trainable head rx
        for index in range(self.n_qubits):
            self.trainable_modules_head_rx[index](self.q_device, wires=index)

        # For the trainable head rz
        for index in range(self.n_qubits):
            self.trainable_modules_head_rz[index](self.q_device, wires=index)

        # For the non-trainable entanglement (add on-the-fly)
        for offset in range(self.n_qubits - 1):
            loc_0 = self.n_qubits - 1 - offset
            loc_1 = loc_0 - 1
            tqf.cx(self.q_device, wires=[loc_0, loc_1], static=self.static_mode, parent_graph=self.graph)

    def get_para_dict(self):
        para_dict = {}
        para_dict['head_rx'] = []
        for rx_module in self.trainable_modules_head_rx:
            for name, parameters in rx_module.named_parameters():   # only one
                para_dict['head_rx'].append(parameters.data.item())

        para_dict['head_rz'] = []
        for rz_module in self.trainable_modules_head_rz:
            for name, parameters in rz_module.named_parameters():   # only one
                para_dict['head_rz'].append(parameters.data.item())

        return para_dict


class QLayer_13(tq.QuantumModule):
    """
    The input to the forward is q_device directly (has the encoded states already)
    The output to the forward is still q_device (has the states after gate)
    """

    def __init__(self, n_qubits):
        super().__init__()
        # gates with trainable parameters
        self.n_qubits = n_qubits

        self.trainable_modules_head = nn.ModuleList()
        for i in range(self.n_qubits):
            self.trainable_modules_head.append(tq.RY(has_params=True, trainable=True))

        self.trainable_modules_head_crz = nn.ModuleList()
        for i in range(self.n_qubits):
            self.trainable_modules_head_crz.append(tq.CRZ(has_params=True, trainable=True))  # The wires not defined here

        self.trainable_modules_middle = nn.ModuleList()
        for i in range(self.n_qubits):
            self.trainable_modules_middle.append(tq.RY(has_params=True, trainable=True))

        self.trainable_modules_middle_crz = nn.ModuleList()
        for i in range(self.n_qubits):
            self.trainable_modules_middle_crz.append(tq.CRZ(has_params=True, trainable=True))  # The wires not defined here

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        """
        1. To convert tq QuantumModule to qiskit or run in the static
        model, need to:
            (1) add @tq.static_support before the forward
            (2) make sure to add
                static=self.static_mode and
                parent_graph=self.graph
                to all the tqf functions, such as tqf.hadamard below
        """
        self.q_device = q_device  # no need to reset here, but get the new states for each batch

        # For the trainable head ry
        for index in range(self.n_qubits):
            self.trainable_modules_head[index](self.q_device, wires=index)

        # For the trainable entanglement crz in the head (wires specified on-the-fly)
        crz_index = 0
        self.trainable_modules_head_crz[crz_index](self.q_device, wires=[self.n_qubits - 1, 0])
        crz_index += 1

        for offset in range(self.n_qubits - 1):
            loc_0 = self.n_qubits - 2 - offset  # control
            loc_1 = loc_0 + 1  # target
            self.trainable_modules_head_crz[crz_index](self.q_device, wires=[loc_0, loc_1])
            crz_index += 1

        assert crz_index == self.n_qubits, "implementation error of first crz in v13"

        # For the trainable middle ry
        for index in range(self.n_qubits):
            self.trainable_modules_middle[index](self.q_device, wires=index)

        # For the trainable entanglement crz in the middle (wires specified on-the-fly)
        crz_index = 0
        f_half_beg = self.n_qubits // 2 + 1  # control_end
        l_half_end = f_half_beg - 1  # control_end

        # build the first half
        for c_index in range(f_half_beg, self.n_qubits):
            loc_0 = c_index  # control
            loc_1 = loc_0 - 1  # target
            self.trainable_modules_middle_crz[crz_index](self.q_device, wires=[loc_0, loc_1])
            crz_index += 1

        # build the last half
        self.trainable_modules_middle_crz[crz_index](self.q_device, wires=[0, self.n_qubits - 1])
        crz_index += 1

        for c_index in range(1, l_half_end + 1):
            loc_0 = c_index  # control
            loc_1 = loc_0 - 1  # target
            self.trainable_modules_middle_crz[crz_index](self.q_device, wires=[loc_0, loc_1])
            crz_index += 1
        # print("In torch model,", crz_index, self.n_qubits)
        assert crz_index == self.n_qubits, "implementation error of last crz in v13"

    def get_para_dict(self):
        para_dict = {}
        para_dict['head'] = []
        for ry_module in self.trainable_modules_head:
            for name, parameters in ry_module.named_parameters():   # only one
                para_dict['head'].append(parameters.data.item())

        para_dict['head_crz'] = []
        for crz_module in self.trainable_modules_head_crz:
            for name, parameters in crz_module.named_parameters():   # only one
                para_dict['head_crz'].append(parameters.data.item())

        para_dict['middle'] = []
        for ry_module in self.trainable_modules_middle:
            for name, parameters in ry_module.named_parameters():   # only one
                para_dict['middle'].append(parameters.data.item())

        para_dict['middle_crz'] = []
        for crz_module in self.trainable_modules_middle_crz:
            for name, parameters in crz_module.named_parameters():   # only one
                para_dict['middle_crz'].append(parameters.data.item())

        return para_dict


class QLayer_19(tq.QuantumModule):
    """
    The input to the forward is q_device directly (has the encoded states already)
    The output to the forward is still q_device (has the states after gate)
    """

    def __init__(self, n_qubits):
        super().__init__()
        # gates with trainable parameters
        self.n_qubits = n_qubits
        self.trainable_modules_head_rx = nn.ModuleList()
        for i in range(self.n_qubits):
            self.trainable_modules_head_rx.append(tq.RX(has_params=True, trainable=True))

        self.trainable_modules_head_rz = nn.ModuleList()
        for i in range(self.n_qubits):
            self.trainable_modules_head_rz.append(tq.RZ(has_params=True, trainable=True))

        self.trainable_modules_crx = nn.ModuleList()
        for i in range(self.n_qubits):
            self.trainable_modules_crx.append(tq.CRX(has_params=True, trainable=True))  # The wires not defined here

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        """
        1. To convert tq QuantumModule to qiskit or run in the static
        model, need to:
            (1) add @tq.static_support before the forward
            (2) make sure to add
                static=self.static_mode and
                parent_graph=self.graph
                to all the tqf functions, such as tqf.hadamard belo
        """
        self.q_device = q_device  # no need to reset here, but get the new states for each batch

        # For the trainable head rx
        for index in range(self.n_qubits):
            self.trainable_modules_head_rx[index](self.q_device, wires=index)

        # For the trainable head rz
        for index in range(self.n_qubits):
            self.trainable_modules_head_rz[index](self.q_device, wires=index)

        # For trainable entanglement (add on-the-fly)
        crx_index = 0
        self.trainable_modules_crx[crx_index](self.q_device, wires=[self.n_qubits - 1, 0])
        crx_index += 1

        for offset in range(self.n_qubits - 1):
            loc_0 = self.n_qubits - 2 - offset  # control
            loc_1 = loc_0 + 1  # target
            self.trainable_modules_crx[crx_index](self.q_device, wires=[loc_0, loc_1])
            crx_index += 1

        assert crx_index == self.n_qubits, "implementation error of crx in v19"

    def get_para_dict(self):
        para_dict = {}
        para_dict['head_rx'] = []
        for rx_module in self.trainable_modules_head_rx:
            for name, parameters in rx_module.named_parameters():   # only one
                para_dict['head_rx'].append(parameters.data.item())

        para_dict['head_rz'] = []
        for rz_module in self.trainable_modules_head_rz:
            for name, parameters in rz_module.named_parameters():   # only one
                para_dict['head_rz'].append(parameters.data.item())

        para_dict['crx'] = []
        for crx_module in self.trainable_modules_crx:
            for name, parameters in crx_module.named_parameters():   # only one
                para_dict['crx'].append(parameters.data.item())

        return para_dict


class QLayer_5(tq.QuantumModule):
    """
    The input to the forward is q_device directly (has the encoded states already)
    The output to the forward is still q_device (has the states after gate)
    """

    def __init__(self, n_qubits):
        super().__init__()
        # gates with trainable parameters
        self.n_qubits = n_qubits

        self.trainable_modules_head_rx = nn.ModuleList()
        for i in range(self.n_qubits):
            self.trainable_modules_head_rx.append(tq.RX(has_params=True, trainable=True))

        self.trainable_modules_head_rz = nn.ModuleList()
        for i in range(self.n_qubits):
            self.trainable_modules_head_rz.append(tq.RZ(has_params=True, trainable=True))

        self.trainable_modules_crz = nn.ModuleList()
        for i in range(self.n_qubits):
            for j in range(self.n_qubits - 1):
                self.trainable_modules_crz.append(tq.CRZ(has_params=True, trainable=True))  # The wires not defined here

        self.trainable_modules_tail_rx = nn.ModuleList()
        for i in range(self.n_qubits):
            self.trainable_modules_tail_rx.append(tq.RX(has_params=True, trainable=True))

        self.trainable_modules_tail_rz = nn.ModuleList()
        for i in range(self.n_qubits):
            self.trainable_modules_tail_rz.append(tq.RZ(has_params=True, trainable=True))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        """
        1. To convert tq QuantumModule to qiskit or run in the static
        model, need to:
            (1) add @tq.static_support before the forward
            (2) make sure to add
                static=self.static_mode and
                parent_graph=self.graph
                to all the tqf functions, such as tqf.hadamard below
        """
        self.q_device = q_device  # no need to reset here, but get the new states for each batch

        # For the trainable head rx
        for index in range(self.n_qubits):
            self.trainable_modules_head_rx[index](self.q_device, wires=index)

        # For the trainable head rz
        for index in range(self.n_qubits):
            self.trainable_modules_head_rz[index](self.q_device, wires=index)

        # For the trainable entanglement crz (wires specified on-the-fly)
        crz_index = 0
        for offset in range(self.n_qubits):  # offset of control end
            loc_0 = self.n_qubits - 1 - offset  # control

            # downside crz gate
            for target_down in range(self.n_qubits - 1, loc_0, -1):
                loc_1 = target_down  # target
                self.trainable_modules_crz[crz_index](self.q_device, wires=[loc_0, loc_1])
                crz_index += 1

            # upside crz gate
            for target_up in range(loc_0 - 1, -1, -1):  # 0 will be included
                loc_1 = target_up  # target
                self.trainable_modules_crz[crz_index](self.q_device, wires=[loc_0, loc_1])
                crz_index += 1

        assert crz_index == self.n_qubits * (self.n_qubits - 1), "implementation error of crz in v5"

        # For the trainable tail rx
        for index in range(self.n_qubits):
            self.trainable_modules_tail_rx[index](self.q_device, wires=index)

        # For the trainable tail rz
        for index in range(self.n_qubits):
            self.trainable_modules_tail_rz[index](self.q_device, wires=index)

    def get_para_dict(self):
        para_dict = {}
        para_dict['head_rx'] = []
        for rx_module in self.trainable_modules_head_rx:
            for name, parameters in rx_module.named_parameters():   # only one
                para_dict['head_rx'].append(parameters.data.item())

        para_dict['head_rz'] = []
        for rz_module in self.trainable_modules_head_rz:
            for name, parameters in rz_module.named_parameters():   # only one
                para_dict['head_rz'].append(parameters.data.item())

        para_dict['crz'] = []
        for crz_module in self.trainable_modules_crz:
            for name, parameters in crz_module.named_parameters():   # only one
                para_dict['crz'].append(parameters.data.item())

        para_dict['tail_rx'] = []
        for rx_module in self.trainable_modules_tail_rx:
            for name, parameters in rx_module.named_parameters():   # only one
                para_dict['tail_rx'].append(parameters.data.item())

        para_dict['tail_rz'] = []
        for rz_module in self.trainable_modules_tail_rz:
            for name, parameters in rz_module.named_parameters():   # only one
                para_dict['tail_rz'].append(parameters.data.item())

        return para_dict


############################################################################################################
# TODO(NOTE): The following is not used
# It is written by hanrui
class QLayer_test(tq.QuantumModule):
    """
    The input to the forward is q_device directly (has the encoded states already)
    """
    def __init__(self):
        super().__init__()

        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(2)])
        self.add_module('conv1', tq.RX(has_params=True, trainable=True))
        self.add_module('conv2', tq.RX(has_params=True, trainable=True))
        self.add_module('conv3', tq.RX(has_params=True, trainable=True))
        self.linears.append(tq.RX(has_params=True, trainable=True))

    @tq.static_support  # what is it used for?
    def forward(self, q_device: tq.QuantumDevice):
        """
        1. To convert tq QuantumModule to qiskit or run in the static
        model, need to:
            (1) add @tq.static_support before the forward
            (2) make sure to add
                static=self.static_mode and
                parent_graph=self.graph
                to all the tqf functions, such as tqf.hadamard below
        """
        self.q_device = q_device  # TODO: In the forward function? dynamic mode? there is no reset for the q_device

        self.random_layer(self.q_device)    # TODO: Why we need it?

        # some trainable gates (instantiated ahead of time)
        self.rx0(self.q_device, wires=0)
        self.ry0(self.q_device, wires=1)
        self.rz0(self.q_device, wires=3)
        self.crx0(self.q_device, wires=[0, 2])

        # add some more non-parameterized gates (add on-the-fly)
        tqf.hadamard(self.q_device, wires=3, static=self.static_mode,
                     parent_graph=self.graph)   # TODO: static mode!!!!
        tqf.sx(self.q_device, wires=2, static=self.static_mode,
               parent_graph=self.graph)
        tqf.cnot(self.q_device, wires=[3, 0], static=self.static_mode,
                 parent_graph=self.graph)

