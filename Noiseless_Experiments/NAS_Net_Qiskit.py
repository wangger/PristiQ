from NAS_dict import qiskit_layer_name_dict
import numpy as np
from qiskit import QuantumCircuit
from output_qiskit import IBM_Q_exe, Noisy_Aer_exe
import torch


def build_VQC_Net_Qiskit(layer_list, layer_para_list, args):
    """
    build the general form of VQC Net in qiskit.
    param: layer_list is the list of layer names (str), which could be used to find the layer class.
           layer_list should be in sequential order
           layer_list is the DNA sequence (hyperparameters)

           layer_para_list: [{para1: value1, para2: value2, ...}, {para1: value1, para2: value2, ...}, {}, ],
           len = n_layers

    NOTE: The qiskit circuit is by default running on the real quantum hareware, we can use pytorch for simulator
    NOTE: The interface of different layers should be same/consistent
    NOTE: It is in wiki order!!!
    """
    n_qubits = args.num_ori_qubits + args.num_enc_qubits  # total qubits

    # TODO(NOTE): Build quantum circuit for QNN model
    model_qc = QuantumCircuit(n_qubits, n_qubits)

    for layer_idx in range(len(layer_list)):
        layer_name = layer_list[layer_idx]
        layer_func = qiskit_layer_name_dict[layer_name]
        layer_para = layer_para_list[layer_idx]  # {para1: value1, para2: value2, ...}

        if layer_func is not None:  # at least has one layer
            layer_qc = layer_func(n_qubits, layer_para)
            model_qc.compose(layer_qc, inplace=True)

    # Add measurement for shot-based execution (It should not be added for statevector simulator)
    tol_q_idx_list = list(range(n_qubits))
    model_qc.measure(tol_q_idx_list, tol_q_idx_list)

    return model_qc


# TODO(Note): Build quantum circuit for encryption, connect it to model qc
def Noisy_Aer_forward(x, enc_model_qc, args):
    """
    Test batch of data

    param:
    x: Assume that the input x is the original input data (bs, c, w, h)
    But it could also handle input like (BS, 2**num_tol_qubits)

    enc_model_qc: the qc consisting of enc_qc and model_qc

    return: (BS, num_output), tensor
    """

    # TODO(NOTE): for the batch data, we can only handle them one by one currently
    # TODO(NOTE): run on real hardware IBMQ
    n_qubits = args.num_ori_qubits + args.num_enc_qubits
    n_enc_qubits = args.num_enc_qubits
    feature_loc = list(range(n_enc_qubits, n_qubits))
    mapping = list(range(n_qubits - 1, -1, -1))  # reverse mapping
    output = []

    for sample in x:
        sample = sample.reshape(-1)  # (2**num_tol_qubits)
        sample = sample.numpy()

        # Build the quantum circuit for data encoding
        sample_qc = QuantumCircuit(n_qubits, n_qubits)  # TODO(Note): need classical qubits here
        sample_qc.initialize(sample, feature_loc)

        # Build the complete quantum circuit
        sample_qc.compose(enc_model_qc, qubits=mapping, inplace=True)

        # TODO: Execute the quantum circuit on noisy Aer simulator, test single data
        sample_output = Noisy_Aer_exe(sample_qc, args)  # (num_output), tensor
        output.append(sample_output)

    output = torch.stack(output, dim=0)
    return output  # (BS, num_output)


# TODO(Note): Build quantum circuit for encryption, connect it to model qc
def IBMQ_forward(x, enc_model_qc, args):
    """
    Test batch of data

    param:
    x: Assume that the input x is the original input data (bs, c, w, h)
    But it could also handle input like (BS, 2**num_tol_qubits)

    enc_model_qc: the qc consisting of enc_qc and model_qc

    return: (BS, num_output), tensor
    """

    # TODO(NOTE): for the batch data, we can only handle them one by one currently
    # TODO(NOTE): run on real hardware IBMQ
    n_qubits = args.num_ori_qubits + args.num_enc_qubits
    n_enc_qubits = args.num_enc_qubits
    feature_loc = list(range(n_enc_qubits, n_qubits))
    mapping = list(range(n_qubits - 1, -1, -1))  # reverse mapping
    output = []

    for sample in x:
        sample = sample.reshape(-1)  # (2**num_tol_qubits)
        sample = sample.numpy()

        # Build the quantum circuit for data encoding
        sample_qc = QuantumCircuit(n_qubits, n_qubits)  # TODO(Note): need classical qubits here
        sample_qc.initialize(sample, feature_loc)

        # Build the complete quantum circuit
        sample_qc.compose(enc_model_qc, qubits=mapping, inplace=True)

        # Execute the quantum circuit on IBMQ, test single data
        sample_output = IBM_Q_exe(sample_qc, args)  # (num_output), tensor
        output.append(sample_output)

    output = torch.stack(output, dim=0)
    return output  # (BS, num_output)


if __name__ == "__main__":
    # Build a vqc in pytorch first (to get the parameters), how to estimate the benchmark
    from NAS_Net import VQC_Net
    # from NAS_main import pareto_args

    class pareto_args():  # for the final retraining of sampled QNN from trained RNN controller
        def __init__(self):
            self.Is_built = True

    pareto_args.num_ori_qubits = 4  # 6  # 4
    pareto_args.num_enc_qubits = 1  # 2
    pareto_args.output_num = 2 # 3
    pareto_args.device = "cpu"

    layer_list = ['v13', 'v10', 'v0', 'v19', 'v10']  # ['v5', 'v5', 'v5', 'v5', 'v5']
    # layer_list = ['v19']  # ['v13']  # ['v5']  # ['v10']  # ['v2']  #
    torch_VQC = VQC_Net(layer_list, pareto_args)

    # get the qiskit model
    layer_list = torch_VQC.get_layer_list()
    layer_para_list = torch_VQC.get_layer_para_list()
    print(torch_VQC.q_layers)
    print(layer_list)
    print(layer_para_list)
    qiskit_VQC = build_VQC_Net_Qiskit(layer_list, layer_para_list, pareto_args)
    cir_len = qiskit_VQC.depth()
    print("circuit length before compilation", cir_len)

    # n_qubits = pareto_args.num_ori_qubits + pareto_args.num_enc_qubits
    # mapping = list(range(n_qubits - 1, -1, -1))  # reverse mapping
    # print(mapping)
    # print(n_qubits)
    # sample_qc = QuantumCircuit(n_qubits, n_qubits)
    # sample_qc.compose(qiskit_VQC, qubits=mapping, inplace=True)
    # qiskit_VQC = sample_qc

    # Draw the circuit before and after transpilation, wiki order, exactly
    from qiskit.tools.visualization import circuit_drawer
    circuit_drawer(qiskit_VQC, filename='circuit_figures.jpg', output='mpl', style={'backgroundcolor': '#EEEEEE'})

    from qiskit import *
    pareto_args.optim_level = 0  # 3
    backend_sim = Aer.get_backend('statevector_simulator')  # qasm_simulator
    # backend_sim = Aer.get_backend('qasm_simulator')  # qasm_simulator
    optim_level = pareto_args.optim_level
    qiskit_model_trans = transpile(qiskit_VQC, backend_sim, optimization_level=optim_level)
    cir_len = qiskit_model_trans.depth()
    print("circuit length after compilation", cir_len)





