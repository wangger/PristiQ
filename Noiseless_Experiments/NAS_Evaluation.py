import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.nn as nn

import torchquantum as tq
import torchquantum.functional as tqf

from NAS_dict import layer_name_dict
from NAS_Net import VQC_Net
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from NAS_train import Training_Ideal, Testing_QC, Testing_Noisy_QC
from NAS_Net_Qiskit import build_VQC_Net_Qiskit
from qiskit import transpile, Aer


def QNN_Evaluation_Ideal(train_loader, valid_loader, QNN, training_args, args, logger, enc_circuit=None):
    """
    Interpret the given DNA to a tf-quantum QNN first,
    then train it on train_set with autograd of pytorch (no qiskit involve here),
    finally evaluate it on the validation set on each training epoch, pick the acc with best epoch
    :param QNN -> nn.module
    :return:
        best_valid_acc < 1
        cir_len is integer
    """
    # # TODO(Note): Interpret the given DNA to a tf-quantum QNN
    # QNN = VQC_Net(n_qubits, n_output, DNA, args)

    # TODO(Note): Train the QNN and return the best validation acc
    # To Get the circuit length, we also need to get the best model for the circuit length
    best_valid_acc, best_epoch, best_model = Training_Ideal(train_loader, valid_loader, QNN, logger, training_args,
                                                            args, enc_circuit)

    # TODO(Note): We only get the length of computation circuit yet.
    # we get the length after transpilation yet
    layer_list = best_model.get_layer_list()
    layer_para_list = best_model.get_layer_para_list()
    # print("The layer list is ", layer_list)
    # print("The layer parameter list is ", layer_para_list)
    qiskit_model = build_VQC_Net_Qiskit(layer_list, layer_para_list, args)
    backend_sim = Aer.get_backend('statevector_simulator')  # qasm_simulator
    optim_level = args.optim_level
    qiskit_model_trans = transpile(qiskit_model, backend_sim, optimization_level=optim_level)
    cir_len = qiskit_model_trans.depth()

    return best_valid_acc, cir_len


def QNN_Evaluation_Noise_Sim(train_loader, valid_loader, QNN, training_args, args, logger, enc_circuit=None,
                             qiskit_enc_circuit=None):

    """
    :return:
        noisy_valid_acc < 1
    """
    # TODO(Note): Train the QNN and return the best validation acc
    # To Get the circuit length, we also need to get the best model for the circuit length
    best_valid_acc, best_epoch, best_model = Training_Ideal(train_loader, valid_loader, QNN, logger, training_args,
                                                            args, enc_circuit)

    # Build the quantum circuit for QNN model
    layer_list = best_model.get_layer_list()
    layer_para_list = best_model.get_layer_para_list()
    qiskit_model = build_VQC_Net_Qiskit(layer_list, layer_para_list, args)

    # TODO(Note): connect the encryption circuit and enc circuit for test
    if qiskit_enc_circuit is not None:
        qiskit_enc_circuit.compose(qiskit_model, inplace=True)
        enc_model_qc = qiskit_enc_circuit
    else:
        enc_model_qc = qiskit_model

    # TODO(Note): Test on the noisy Aer simulator
    noisy_valid_acc = Testing_Noisy_QC(enc_model_qc, valid_loader, logger, args)

    return noisy_valid_acc


def QNN_Evaluation_Noise_QC(train_loader, valid_loader, QNN, training_args, args, logger, enc_circuit=None,
                            qiskit_enc_circuit=None):
    """
    :return:
        noisy_valid_acc < 1
    """
    # TODO(Note): Train the QNN and return the best validation acc
    # To Get the circuit length, we also need to get the best model for the circuit length
    best_valid_acc, best_epoch, best_model = Training_Ideal(train_loader, valid_loader, QNN, logger, training_args,
                                                            args, enc_circuit)

    # Build the quantum circuit for QNN model
    layer_list = best_model.get_layer_list()
    layer_para_list = best_model.get_layer_para_list()
    qiskit_model = build_VQC_Net_Qiskit(layer_list, layer_para_list, args)

    # TODO(Note): connect the encryption circuit and enc circuit for test
    if qiskit_enc_circuit is not None:
        qiskit_enc_circuit.compose(qiskit_model, inplace=True)
        enc_model_qc = qiskit_enc_circuit
    else:
        enc_model_qc = qiskit_model

    # Test the real qc in IBMQ
    noisy_valid_acc = Testing_QC(enc_model_qc, valid_loader, logger, args)

    return noisy_valid_acc


# if __name__ == "__main__":
#     from NAS_para import parse_args
#     from utils import fix_random_seeds
#     args = parse_args()
#     print("=" * 21, "Your setting is listed as follows", "=" * 22)
#     print("\t{:<25} {:<15}".format('Attribute', 'Input'))
#     for k, v in vars(args).items():
#         if v is not None:
#             v = str(v)
#             print("\t{:<25} {:<15}".format(k, v))
#     print("=" * 22, "Exploration will start, have fun", "=" * 22)
#     print("=" * 78)
#
#     # TODO: Set the random seed
#     fix_random_seeds(args.seed)
#
#     # TODO(NOTE): Decode the input parameters
#     has_cuda = torch.cuda.is_available()
#     device = torch.device(args.device if has_cuda else "cpu")
#     print("The program is running at {}".format(device))
#     args.device = device  # to make device and args.device consistent, useful for the encoding
#
#     # TODO(NOTE): Load dataset
#     # For mnist dataset
#     from c_input import *
#     interest_class = args.interest_class
#     train_loader, test_loader = load_data(interest_class, args)
#     # TODO: split train_loader to train_loader and valid_loader
#
#     DNA = []  # TODO: get the DNA
#     # TODO: Get the other parameters
#
#     # use test set temporarily
#     best_valid_acc, cir_len = QNN_Evaluation_Ideal(train_loader, valid_loader, DNA, n_qubits, n_output, args,
#                                                    logger, enc_circuit=None)
#
#     # TODO: Search Algorithm
#
#     # TODO: evaluate the trained model on test set


