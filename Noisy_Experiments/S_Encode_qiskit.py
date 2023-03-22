import numpy as np
import sys
from utils import *
import math
from qiskit import QuantumCircuit


def build_enc_circuit(args):
    """
    It is still in wiki order! So the qubit is added at the bottom
    """
    num_ori_qubits = args.num_ori_qubits
    num_enc_qubits = args.num_enc_qubits
    num_tol_qubits = num_ori_qubits + num_enc_qubits
    start_loc = num_ori_qubits  # start index of added qubit

    pre_gate = args.pre_gate  # string
    if args.ry_angle_factor_list is not None:
        theta_list = [theta * math.pi for theta in args.ry_angle_factor_list]
    else:
        theta_list = None

    entag_pattern = args.entag_pattern  # string
    permutation_list = args.permutation_list   # It could be None

    # TODO(NOTE): start to build the quantum circuit
    qc = QuantumCircuit(num_tol_qubits, num_tol_qubits)  # TODO(NOTE): we need classical qubit! verify!

    # Add some more non-parameterized gates (add on-the-fly) in wiki order!!!!
    # copy + scale
    if pre_gate == 'ry':
        for offset, theta in enumerate(theta_list):    # theta_list cannot be None
            qc.ry(theta, start_loc + offset)
    elif pre_gate == 'hadamard':
        for index in range(start_loc, num_tol_qubits):
            qc.h(index)
    elif pre_gate == 'identity':
        pass  # No scaling operation for the padding input
    else:
        raise Exception("The gate is not supported by the encoding circuit!")

    # permutation part
    if entag_pattern == 'random':
        for cnot_wires in permutation_list:  # cannot be None
            index_c, index_t = cnot_wires

            qc.cx(index_c, index_t)

    elif entag_pattern == 'single':
        # TODO (NOTE): The index setting is critical
        offset = num_ori_qubits    # used for target qubit loc
        for index in range(start_loc, num_tol_qubits):
            # This is for a single CNOT
            index_c = index  # loc of the added qubit
            index_t = index_c - offset  # The original qubit is controlled by the added qubit
            qc.cx(index_c, index_t)

    elif entag_pattern == 'full':
        for c_index in range(start_loc, num_tol_qubits):
            index_c = c_index  # loc of the added qubit
            for t_index in range(num_ori_qubits):  # entangle each added qubit with all of the original qubits
                index_t = t_index  # Each original qubit is controlled by the added qubit
                qc.cx(index_c, index_t)

    elif entag_pattern == 'identity':
        pass  # No permutation operation for the copied input

    elif entag_pattern == 'single_add_0':
        for i in range(0, num_ori_qubits, 2):  # the depth is only 1
            if i > num_ori_qubits - 2:
                break
            else:
                qc.cx(i, i+1)

        # TODO (NOTE): The index setting is critical
        offset = num_ori_qubits    # used for target qubit loc
        for index in range(start_loc, num_tol_qubits):
            # This is for a single CNOT
            index_c = index  # loc of the added qubit
            index_r = index_c - offset  # The original qubit is controlled by the added qubit
            qc.cx(index_c, index_r)

    elif entag_pattern == 'single_add_1':
        for i in range(0, num_ori_qubits-1):  # the depth is only 1
            qc.cx(i, i + 1)
        # TODO (NOTE): The index setting is critical
        offset = num_ori_qubits  # used for target qubit loc
        for index in range(start_loc, num_tol_qubits):
            # This is for a single CNOT
            index_c = index  # loc of the added qubit
            index_r = index_c - offset  # The original qubit is controlled by the added qubit
            qc.cx(index_c, index_r)
    else:
        raise Exception("This type of permutation is not currently supported!")

    return qc



