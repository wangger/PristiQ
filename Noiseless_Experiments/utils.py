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
    ori_img: (BS, 2**num_qubits)
    enc_img: (BS, 2**num_qubits)
    """
    bsz = ori_img.shape[0]
    n_pixels = ori_img.shape[1]
    sq = (ori_img - enc_img)**2  # (BS, 2**num_qubits)
    bsz_mse = (1/n_pixels) * torch.sum(sq, dim=1)  # (BS)
    avg_mse = (1/bsz) * torch.sum(bsz_mse, dim=0)  # (1)
    bsz_PSNR = 10 * torch.log10(1/bsz_mse)
    avg_PSNR = (1/bsz) * torch.sum(bsz_PSNR, dim=0)   # (1)

    return bsz_PSNR, avg_PSNR, bsz_mse, avg_mse


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