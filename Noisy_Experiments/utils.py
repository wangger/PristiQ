import torch
import math
import numpy as np
import random


def set_value(mm, col, row, val):  # TODO(NOTE): it could be traced by autograd
    index = (torch.LongTensor([col]), torch.LongTensor([row]))  # Generate the index
    # print(index)
    # print(val)
    mm = mm.index_put(index, val)
    return mm


def qf_sum(n_qubits):
    sum_mat = []
    flag = "0" + str(n_qubits) + "b"
    for i in range(0, int(math.pow(2, n_qubits))):
        bit_str = format(i, flag)
        row = []
        for c in bit_str:
            row.append(float(c))
        sum_mat.append(row)
    return sum_mat


def amp2prop(state):  # state (# amplitude , bs)
    state = state.double()  # change the data type
    state = state * state   # TODO(Note): get the probability for each basis, use torch.mul?
    n_qubits = int(math.log2(state.shape[0]))
    sum_mat = torch.tensor(qf_sum(n_qubits), dtype=torch.float64)
    # print("=========Test amplitude============")
    # print(sum_mat.shape)  # [# amplitude, # qubit]
    # print(sum_mat)
    
    sum_mat = sum_mat.t()  # [# qubit, # amplitude]
    state = torch.mm(sum_mat, state)  # [# qubit, bs]
    # print("=========Test state============")
    # sum = state[0, 0] + state[0, 0]
    # print(state)
    return state


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # set seed for torchvision.transforms
    random.seed(seed)

    print("set random seed: {:}".format(seed))


def cal_PSNR(ori_img, enc_img):
    """
    calculate the PSNR of a batch data
    :param
    ori_img: (BS, 2**num_qubits)
    enc_img: (BS, 2**num_qubits)
    :return

    """
    bsz = ori_img.shape[0]
    n_pixels = ori_img.shape[1]
    sq = (ori_img - enc_img)**2  # (BS, 2**num_qubits)
    bsz_mse = (1/n_pixels) * torch.sum(sq, dim=1)  # (BS)
    avg_mse = (1/bsz) * torch.sum(bsz_mse, dim=0)  # (1)
    bsz_PSNR = 10 * torch.log10(1/bsz_mse)
    avg_PSNR = (1/bsz) * torch.sum(bsz_PSNR, dim=0)   # (1)

    return bsz_PSNR, avg_PSNR, bsz_mse, avg_mse


# [NOTE]: It should not be used
def get_cir_len_baseline(args):
    from NAS_Net import VQC_Net
    layer_list = ['v5', 'v5', 'v5', 'v5', 'v5']
    torch_VQC = VQC_Net(layer_list, args)

    layer_list = torch_VQC.get_layer_list()
    layer_para_list = torch_VQC.get_layer_para_list()
    from NAS_Net_Qiskit import build_VQC_Net_Qiskit
    qiskit_VQC = build_VQC_Net_Qiskit(layer_list, layer_para_list, args)
    cir_len_bt = qiskit_VQC.depth()

    from qiskit import Aer, transpile
    backend_sim = Aer.get_backend('statevector_simulator')  # 'qasm_simulator'
    optim_level = 0
    qiskit_model_trans = transpile(qiskit_VQC, backend_sim, optimization_level=optim_level)
    cir_len_at = qiskit_model_trans.depth()

    return cir_len_at


if __name__ == "__main__":
    import pandas as pd
    csv_path = "Baseline_Readin/mnist_3,6_baseline_input.csv"  # "read_test.csv"
    df = pd.read_csv(csv_path)
    print(df)
    print(df["NAS_best_acc_arch"])
    # print(df["num_enc_qubits"])
    # print(df["ry_angle_factor_list"])
    # print(df["entag_pattern"])
    # print(df["permutation_list"])

    # In ORC file
    test = None
    for row in df["reward_beta"]:  # int or float
        print(type(row))
        print(row)
    print("==========reward_beta========================")

    # In ORC file
    test = None
    for row in df["NAS_best_acc_arch"]:
        print(type(row))
        print(row)
        row = ",".join(row.split())
        print(row)  # v2,v0,v2,v5,v5
        test = row
        break

    # In program
    test = [layer for layer in test.split(",")]
    print(test)
    print("-"*20, "Model architecture", "-"*20)

    # In ORC file
    for row in df["num_enc_qubits"]:    # int # TODO: convert type
        print(type(row))
        print(row)

    # In ORC file
    for row in df["ry_angle_factor_list"]:
        print(type(row))
        print(row)
        row = row.strip("[").strip("]").replace(" ", "")
        print(row)

    # In program
    # if args.ry_angle_factor_list is not None:
    #     ry_angle_factor_list = [float(x.strip()) for x in args.ry_angle_factor_list.split(",")]
    #     print(ry_angle_factor_list)
    #     args.ry_angle_factor_list = ry_angle_factor_list

    print("-" * 20, "ry_angle_factor_list", "-" * 20)

    for row in df["entag_pattern"]:
        print(type(row))
        print(row)

    # import re
    # def strip(text, chars=None):
    #     if chars is None:
    #         reg = re.compile('^ *| *$')
    #     else:
    #         reg = re.compile('^[' + chars + ']*|[' + chars + ']*$')
    #     return reg.sub('', text)

    # for row in df["permutation_list"]:
    #     print(type(row))
    #     print(row)
    #     row = [x.strip(' [').strip('[').strip(']').replace(" ", "") for x in row.split("],")]
    #     # row = [x.strip("[").strip("]") for x in row.split(",")]
    #     print(row)
    #     row = [[int(y) for y in x.split(",")] for x in row]
    #     print(row)

    # In ORC file
    test = None
    for row in df["permutation_list"]:
        print(type(row))
        print(row)
        if row != "None":
            row = [x.strip(' [').strip('[').strip(']').replace(" ", "") for x in row.split("],")]
            # row = [x.strip("[").strip("]") for x in row.split(",")]
            print(row)
            print(type(row))
            row = str(row)
            print(row)
            print(type(row))
            print(row[0])
            row = row.replace(" ", "")
            print(row)
            test = row
            break

    # In the baseline program
    print("I am here")
    print(test)
    print(test.strip('[').strip(']').split('\',\''))
    test = [[int(y.strip("\'")) for y in x.split(",")] for x in test.strip('[').strip(']').split('\',\'')]
    print("I am here")
    print(test)
    print(test[0])