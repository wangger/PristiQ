# from lib_qc import *
# from lib_util import *
# from lib_net import *
import argparse
import time
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import os
import sys
sys.path.append("../interfae/")
# from lib_model_summary import summary
from collections import Counter
from pathlib import Path
# from qiskit_simulator_wbn import run_simulator
from S_Encode import Encoding_Circuit
from utils import fix_random_seeds, cal_PSNR
from c_input import *
import math
import matplotlib.pyplot as plt
import torchvision
import logging
import torchquantum.functional as tqf
import torchquantum as tq
from torchquantum.macro import C_DTYPE
from S_Encode import kronecker_product_einsum_batched

logging.basicConfig(stream=sys.stdout,
                    level=logging.WARNING,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


def PSNR_Baseline_cal(test_loader_clean, test_loader_amp, ori_enc_circuit, args):
    base_total_PSNR = 0
    base_total_MSE = 0
    base_total_data_size = 0

    device = args.device

    test_loader_clean_iter = iter(test_loader_clean)  # create once is enough!!!
    for batch_idx, (amp_images, _) in enumerate(test_loader_amp):
        try:
            clean_images, _ = next(test_loader_clean_iter)
        except StopIteration:
            raise Exception("Load data error!")

        # print(clean_images.shape)
        # print(amp_images.shape)

        clean_images = clean_images.to(device)
        amp_images = amp_images.to(device)
        bsz = clean_images.shape[0]
        # print(bsz)

        clean_images_fla = ori_enc_circuit(clean_images)  # [BS, 2 ** num_tol_qubits]
        clean_images_fla = clean_images_fla.to(torch.float32)
        amp_images_fla = ori_enc_circuit(amp_images)  # [BS, 2 ** num_tol_qubits]
        amp_images_fla = amp_images_fla.to(torch.float32)

        # print("-" * 30, "start to calculate the PSNR", "-" * 30)
        bsz_PSNR, avg_PSNR, bsz_mse, avg_mse = cal_PSNR(clean_images_fla,
                                                        amp_images_fla)  # The function is for one batch

        base_total_PSNR += avg_PSNR * bsz
        base_total_MSE += avg_mse * bsz
        base_total_data_size += bsz

    base_final_PSNR = base_total_PSNR / base_total_data_size
    base_final_MSE = base_total_MSE / base_total_data_size

    print("The PSNR of baseline is {}".format(base_final_PSNR))
    print("The MSE of baseline is {}".format(base_final_MSE))

    return base_final_PSNR, base_final_MSE


def PSNR_Key_cal(test_loader_clean, test_loader_amp, ori_enc_circuit, enc_circuit, args):
    total_PSNR = 0
    total_MSE = 0
    total_data_size = 0

    device = args.device

    test_loader_clean_iter = iter(test_loader_clean)  # reset the iterator

    for batch_idx, (amp_images, _) in enumerate(test_loader_amp):
        try:
            clean_images, _ = next(test_loader_clean_iter)
        except StopIteration:
            raise Exception("Load data error!")

        clean_images = clean_images.to(device)
        amp_images = amp_images.to(device)
        bsz = clean_images.shape[0]

        clean_images_fla = ori_enc_circuit(clean_images)  # [BS, 2 ** num_tol_qubits]
        clean_images_fla = clean_images_fla.to(torch.float32)
        se_amp_images_fla = enc_circuit(amp_images)  # [BS, 2 ** num_tol_qubits]
        se_amp_images_fla = se_amp_images_fla.to(torch.float32)

        # print("-" * 30, "start to calculate the PSNR", "-" * 30)
        bsz_PSNR, avg_PSNR, bsz_mse, avg_mse = cal_PSNR(clean_images_fla,
                                                        se_amp_images_fla)  # The function is for one batch

        total_PSNR += avg_PSNR * bsz
        total_MSE += avg_mse * bsz
        total_data_size += bsz

    final_PSNR = total_PSNR / total_data_size
    final_MSE = total_MSE / total_data_size
    ry_angle_factor_list = args.ry_angle_factor_list
    permutation_list = args.permutation_list
    entag_pattern = args.entag_pattern
    print("The rotation angles are {}".format(ry_angle_factor_list))
    print("The permutation_list are {}".format(permutation_list))
    print("The entangle pattern is {}".format(entag_pattern))
    print("The current final PSNR is {}".format(final_PSNR))
    print("The current final MSE is {}".format(final_MSE))

    return final_PSNR, final_MSE