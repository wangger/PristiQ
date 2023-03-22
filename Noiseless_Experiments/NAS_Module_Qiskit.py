import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.nn as nn

import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit

# TODO(Note): The interface of layer should be consistent (in the same form)!


# Zhirui's Model (ry + circuit 10)
def build_QLayer_10(n_qubits, layer_para):
    """
    param: layer_para is a dict
    NOTE: It is in WIKI order!!!
    """
    qc = QuantumCircuit(n_qubits)   # TODO: do not need classical qubit? verify!
    head_para = layer_para['head']  # list
    tail_para = layer_para['tail']  # list

    # For the trainable head
    for index in range(n_qubits):
        theta = head_para[index]
        qc.ry(theta, index)

    # For the non-trainable entanglement
    for offset in range(n_qubits - 1):
        loc_0 = n_qubits - 1 - offset
        loc_1 = loc_0 - 1
        qc.cz(loc_0, loc_1)

    qc.cz(0, n_qubits - 1)

    # For the trainable tail
    for index in range(n_qubits):
        theta = tail_para[index]
        qc.ry(theta, index)

    return qc


def build_QLayer_2(n_qubits, layer_para):
    """
    param: layer_para is a dict
    NOTE: It is in WIKI order!!!
    """
    qc = QuantumCircuit(n_qubits)
    head_rx_para = layer_para['head_rx']  # list
    head_rz_para = layer_para['head_rz']  # list

    # For the trainable head rx
    for index in range(n_qubits):
        theta = head_rx_para[index]
        qc.rx(theta, index)

    # For the trainable head rz
    for index in range(n_qubits):
        theta = head_rz_para[index]
        qc.rz(theta, index)

    # For the non-trainable entanglement (add on-the-fly)
    for offset in range(n_qubits - 1):
        loc_0 = n_qubits - 1 - offset
        loc_1 = loc_0 - 1
        qc.cx(loc_0, loc_1)

    return qc


def build_QLayer_13(n_qubits, layer_para):
    """
    param: layer_para is a dict
    NOTE: It is in WIKI order!!!
    """
    qc = QuantumCircuit(n_qubits)
    head_para = layer_para['head']  # list
    head_crz_para = layer_para['head_crz']  # list
    middle_para = layer_para['middle']  # list
    middle_crz_para = layer_para['middle_crz']  # list

    # For the trainable head ry
    for index in range(n_qubits):
        theta = head_para[index]
        qc.ry(theta, index)

    # For the trainable entanglement crz in the head
    crz_index = 0
    crz_theta = head_crz_para[crz_index]
    qc.crz(crz_theta, n_qubits - 1, 0)
    crz_index += 1
    for offset in range(n_qubits - 1):
        loc_0 = n_qubits - 2 - offset  # control
        loc_1 = loc_0 + 1  # target
        crz_theta = head_crz_para[crz_index]
        qc.crz(crz_theta, loc_0, loc_1)
        crz_index += 1

    assert crz_index == n_qubits, "qiskit implementation error of first crz in v13"

    # For the trainable middle ry
    for index in range(n_qubits):
        theta = middle_para[index]
        qc.ry(theta, index)

    # For the trainable entanglement crz in the middle (wires specified on-the-fly)
    crz_index = 0
    f_half_beg = n_qubits // 2 + 1  # control_end
    l_half_end = f_half_beg - 1  # control_end

    # build the first half
    for c_index in range(f_half_beg, n_qubits):
        loc_0 = c_index  # control
        loc_1 = loc_0 - 1  # target
        crz_theta = middle_crz_para[crz_index]
        qc.crz(crz_theta, loc_0, loc_1)
        crz_index += 1

    # build the last half
    crz_theta = middle_crz_para[crz_index]
    qc.crz(crz_theta, 0, n_qubits - 1)
    crz_index += 1

    for c_index in range(1, l_half_end + 1):
        loc_0 = c_index  # control
        loc_1 = loc_0 - 1  # target
        crz_theta = middle_crz_para[crz_index]
        qc.crz(crz_theta, loc_0, loc_1)
        crz_index += 1

    # print("In qiskit model,", crz_index, n_qubits)
    assert crz_index == n_qubits, "qiskit implementation error of last crz in v13"

    return qc


def build_QLayer_19(n_qubits, layer_para):
    """
    param: layer_para is a dict
    NOTE: It is in WIKI order!!!
    """
    qc = QuantumCircuit(n_qubits)
    # print(layer_para.keys())
    head_rx_para = layer_para['head_rx']  # list
    head_rz_para = layer_para['head_rz']  # list
    crx_para = layer_para['crx']  # list

    # For the trainable head rx
    for index in range(n_qubits):
        theta = head_rx_para[index]
        qc.rx(theta, index)

    # For the trainable head rz
    for index in range(n_qubits):
        theta = head_rz_para[index]
        qc.rz(theta, index)

    # For trainable entanglement (add on-the-fly)
    crx_index = 0
    crx_theta = crx_para[crx_index]
    qc.crx(crx_theta, n_qubits - 1, 0)
    crx_index += 1

    for offset in range(n_qubits - 1):
        loc_0 = n_qubits - 2 - offset  # control
        loc_1 = loc_0 + 1  # target
        crx_theta = crx_para[crx_index]
        qc.crx(crx_theta, loc_0, loc_1)
        crx_index += 1

    assert crx_index == n_qubits, "qiskit implementation error of crx in v19"

    return qc


def build_QLayer_5(n_qubits, layer_para):
    """
    param: layer_para is a dict
    NOTE: It is in WIKI order!!!
    """
    qc = QuantumCircuit(n_qubits)
    head_rx_para = layer_para['head_rx']  # list
    head_rz_para = layer_para['head_rz']  # list
    crz_para = layer_para['crz']  # list
    tail_rx_para = layer_para['tail_rx']
    tail_rz_para = layer_para['tail_rz']

    # For the trainable head rx
    for index in range(n_qubits):
        theta = head_rx_para[index]
        qc.rx(theta, index)

    # For the trainable head rz
    for index in range(n_qubits):
        theta = head_rz_para[index]
        qc.rz(theta, index)

    # For the trainable entanglement crz
    crz_index = 0
    for offset in range(n_qubits):  # offset of control end
        loc_0 = n_qubits - 1 - offset  # control

        # downside crz gate
        for target_down in range(n_qubits - 1, loc_0, -1):
            loc_1 = target_down  # target
            crz_theta = crz_para[crz_index]
            qc.crz(crz_theta, loc_0, loc_1)
            crz_index += 1

        # upside crz gate
        for target_up in range(loc_0 - 1, -1, -1):  # 0 will be included
            loc_1 = target_up  # target
            crz_theta = crz_para[crz_index]
            qc.crz(crz_theta, loc_0, loc_1)
            crz_index += 1

    assert crz_index == n_qubits * (n_qubits - 1), "qiskit implementation error of crz in v5"

    # For the trainable tail rx
    for index in range(n_qubits):
        theta = tail_rx_para[index]
        qc.rx(theta, index)

    # For the trainable tail rz
    for index in range(n_qubits):
        theta = tail_rz_para[index]
        qc.rz(theta, index)

    return qc