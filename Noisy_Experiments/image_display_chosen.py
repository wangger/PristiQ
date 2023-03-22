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
from c_input import select_num, ToQuantumData
# from Module import Net
from S_Encode import Encoding_Circuit
from utils import fix_random_seeds
import math
import matplotlib.pyplot as plt
import torchvision
from SelfMNIST import *
from c_input import *
from PSNR_calculation import Ori_encoding_args, PSNR_Encoding_Circuit
from utils import cal_PSNR
from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib

import logging
logging.basicConfig(stream=sys.stdout,
                    level=logging.WARNING,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


# def save_image(
#     tensor: Union[torch.Tensor, List[torch.Tensor]],
#     fp: Union[Text, pathlib.Path, BinaryIO],
#     format: Optional[str] = None,
#     **kwargs
# ) -> None:
#     grid = torchvision.utils.make_grid(tensor, **kwargs)
#     # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
#     ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#     im = Image.fromarray(ndarr)
#     # im.save(fp, format=format)
#     im.save(fp, format=format, cmap=plt.cm.hot)
#     print("Save image called!")

def save_image(images, fig_num, columns, PSNR, path):
    fig = plt.figure()
    for i in range(images.shape[0]):
        plt.subplot(int(fig_num/columns), columns, i + 1)
        plt.tight_layout()
        # print(i, images[i][0].shape)
        plt.imshow(images[i][0], cmap='viridis', interpolation='none')
        plt.title("PSNR: {} ".format(PSNR[i]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig(path)   # ("squares.jpg")


def save_image_original(images, fig_num, columns, Label, path):
    fig = plt.figure()
    for i in range(images.shape[0]):
        plt.subplot(int(fig_num/columns), columns, i + 1)
        plt.tight_layout()
        # print(i, images[i][0].shape)
        plt.imshow(images[i][0], cmap='viridis', interpolation='none')
        plt.title("Label: {}".format(Label[i]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig(path)   # ("squares.jpg")


# def load_data(interest_num, args, is_to_q=True, is_shuffle=False):
#     # Get the necessary input parameters
#     img_size = int(args.img_size)
#     batch_size = int(args.batch_size)
#     num_workers = int(args.num_workers)
#     inference_batch_size = int(args.inference_batch_size)
#     isppd = args.preprocessdata
#     datapath = args.datapath
#
#     if isppd:
#         train_data = SelfMNIST(root=datapath, img_size=img_size, train=True)
#         test_data = SelfMNIST(root=datapath, img_size=img_size, train=False)
#
#     else:
#         # Convert data to torch.FloatTensor
#         if is_to_q:
#             # TODO (NOTE): encode the data by amplitude encoding
#             # TODO: Why does the input data has no normalization?? since the resize?
#             transform = transforms.Compose(
#                 [transforms.Resize((img_size, img_size)), transforms.ToTensor(), ToQuantumData(img_size)])
#             transform_inference = transforms.Compose(
#                 [transforms.Resize((img_size, img_size)), transforms.ToTensor(), ToQuantumData(img_size)])
#
#         else:
#             transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
#             transform_inference = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
#
#         # choose the training and test datasets
#         train_data = datasets.MNIST(root=datapath, train=True,
#                                     download=True, transform=transform)
#         test_data = datasets.MNIST(root=datapath, train=False,
#                                    download=True, transform=transform_inference)
#
#     # the label of train_data and test_data is consistent
#     # TODO(NOTE): Comment it for convenience
#     # train_data = select_num(train_data, interest_num)  # TODO: Highlight
#     # test_data = select_num(test_data, interest_num)
#
#     # prepare data loaders
#     # TODO: Why drop last?
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
#                                                num_workers=num_workers, shuffle=is_shuffle, drop_last=True)
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size=inference_batch_size,
#                                               num_workers=num_workers, shuffle=False, drop_last=True)
#
#     return train_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser(description='QuantumFlow Classification Training')
    parser.add_argument('--seed', type=int, default=2, help='random seed')  # use cuda to specify GPU

    # Data related
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'fmnist'],
                        help='The dataset used for this experiment')
    parser.add_argument('--device', type=str, default='cpu', help='device')  # use 'cuda' to specify GPU
    parser.add_argument('-c', '--interest_class', type=str, default="3, 6", help="investigate classes")
    parser.add_argument('-s', '--img_size', type=int, default=4, help="image size 4: 4*4")
    parser.add_argument('-j', '--num_workers', type=int, default=0, help="worker to load data", )
    parser.add_argument('-tb', '--batch_size', type=int, default=32, help="training batch size", )
    parser.add_argument('-ib', '--inference_batch_size', type=int, default=256, help="inference batch size", )
    parser.add_argument('-ppd', "--preprocessdata", action="store_true", help="Using the preprocessed data")

    # File
    parser.add_argument('-chk', "--save_chkp", action="store_true", help="Save checkpoints")
    parser.add_argument('-chkname', '--chk_name', type=str, default='', help='folder name for chkpoint')
    # parser.add_argument("--save_path", help="save path", )
    parser.add_argument('-dp', '--datapath', type=str, default='pytorch/data/',
                        help='root path of the dataset of mnist/fmnist')

    # TODO(NOTE): Security encoding related
    parser.add_argument("--Is_S_Encoding", help="Whether we want to do security encoding", action="store_true")
    parser.add_argument("--num_ori_qubits", type=int, default=4, help="Number of original qubits")
    parser.add_argument("--num_enc_qubits", type=int, default=0, help="Number of encoding qubits")

    # The following parameters are valid only when args.Is_S_Encoding is set
    parser.add_argument('--pre_gate', default='ry', choices=['identity', 'hadamard', 'ry'],
                        help='The gate for added qubits before entanglement (default=hadamard)')
    # 0.33,0.67
    parser.add_argument('--ry_angle_factor_list', type=str, default=None, help='The list of angle factors (of pi) for '
                                                                               'added qubits')
    parser.add_argument('--entag_pattern', default='single', type=str, help='The pattern of entanglement for added '
                                                                            'qubits (default=single).')  # choices=['identity', 'single', 'full'],
    # parser.add_argument('--ry_angle_factor_list', nargs='+', type=float, default=None,
    #                     help='The list of angle factors (of pi) for added qubits')
    # parser.add_argument('--ry_angle_factor_list_1', nargs='+', type=float, default=None,
    #                     help='The list of angle factors (of pi) for added qubits')
    # parser.add_argument('--ry_angle_factor_list_2', nargs='+', type=float, default=None,
    #                     help='The list of angle factors (of pi) for added qubits')
    # parser.add_argument('-deb', "--debug", action="store_true", help="Debug mode")
    parser.add_argument("--permutation_list", type=str, default=None, help="given permutation list")

    # TODO(NOTE): Used for PSNR comparison
    parser.add_argument("--max_num_enc_qubits", type=int, default=0, help="Number of maximum encoding qubits for "
                                                                          "comparison")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("=" * 100)
    print("Training procedure for Quantum Computer:")
    print("\tStart at:", time.strftime("%m/%d/%Y %H:%M:%S"))
    print("\tProblems and issues, please contact Dr. Weiwen Jiang (wjiang2@nd.edu)")
    print("\tEnjoy and Good Luck!")
    print("=" * 100)
    print()

    args = parse_args()
    # Set the random seed
    seed = args.seed
    fix_random_seeds(seed)
    ry_angle_list_set = []
    # TODO: Add
    ry_angle_list_set.append([0.33])  # must add a list!
    ry_angle_list_set.append([0.33, 0.67])  # must add a list!

    # TODO(NOTE): Decode the input parameters
    # chkpath = args.chk_path
    has_cuda = torch.cuda.is_available()
    device = torch.device(args.device if has_cuda else "cpu")
    print("The program is running at {}".format(device))
    args.device = device  # to make device and args.device consistent, useful for the encoding

    # data related
    # interest_class = [int(x.strip()) for x in args.interest_class.split(",")]  # for -c
    n_class = 10    # TODO(NOTE): Fixed for our experiment
    dataset = args.dataset
    interest_class = list(range(n_class))  # customize for 10-class
    args.interest_class = interest_class
    args.preprocessdata = False  # TODO(Note): It should be always false
    img_size = args.img_size

    # TODO(Note): Parameters related to security
    Is_S_Encoding = args.Is_S_Encoding
    num_ori_qubits = args.num_ori_qubits
    num_enc_qubits = args.num_enc_qubits
    max_num_enc_qubits = args.max_num_enc_qubits
    args.pre_gate = 'ry'  # TODO(Note): Always ry
    entag_pattern = args.entag_pattern  # string

    # TODO(Note): Pi should be multiplied in the following program
    print(args.ry_angle_factor_list)
    if args.ry_angle_factor_list is not None:
        ry_angle_factor_list = [float(x.strip()) for x in args.ry_angle_factor_list.split(",")]
        print(ry_angle_factor_list)
        args.ry_angle_factor_list = ry_angle_factor_list

    save_chkp = args.save_chkp  # for -chk

    print("=" * 21, "Your setting is listed as follows", "=" * 22)
    print("\t{:<25} {:<15}".format('Attribute', 'Input'))
    for k, v in vars(args).items():
        if v is not None:
            v = str(v)
            print("\t{:<25} {:<15}".format(k, v))
    print("=" * 22, "Exploration will start, have fun", "=" * 22)
    print("=" * 78)

    # Load data set without unitary encoding (but downsample)
    if dataset == "mnist":
        args.output_num = len(interest_class)   # must be 10
        _, test_loader_ori = load_data_mnist(interest_class, args, is_to_q=False, is_shuffle=False,
                                             disable_visualize=False)
    elif dataset == "fmnist":
        args.output_num = len(interest_class)
        _, test_loader_ori = load_data_fmnist(interest_class, args, is_to_q=False, is_shuffle=False,
                                              disable_visualize=False)
    else:
        raise Exception("The dataset is not supported!")

    # TODO(Note): Save the batch of original images
    images_ori, labels_ori = next(iter(test_loader_ori))
    print("image shape: ", images_ori.shape)
    print(images_ori[0].shape)
    # torchvision.utils.save_image(images_ori[0], 'images_no_amp_{}.jpg'.format(img_size), cmap='gist_rainbow')
    # save_image(images_ori[0], 'images_no_amp_{}.jpg'.format(img_size))


    # plt.set_cmap('gist_rainbow')
    # plt.plot()
    # grid_img = torchvision.utils.make_grid(images_ori, nrow=8)
    # plt.imshow(grid_img.permute(1, 2, 0), cmap='gist_rainbow')

    # plt.plot(0)
    # # plt.imshow(images_ori[0])
    # images_ori[0] = images_ori[0].unsqueeze(-1)
    # print(images_ori[0].shape)
    # plt.imshow(images_ori[0], cmap='gist_rainbow')
    # plt.savefig("test_color.jpg", dpi=1200)

    bsz = images_ori.shape[0]
    image_c = images_ori.shape[1]   # channel
    image_h = images_ori.shape[2]

    # Load data set with amplitude encoding (but downsample)
    if dataset == "mnist":
        args.output_num = len(interest_class)   # must be 10
        _, test_loader_amp = load_data_mnist(interest_class, args, is_to_q=True, is_shuffle=False,
                                             disable_visualize=False)
    elif dataset == "fmnist":
        args.output_num = len(interest_class)
        _, test_loader_amp = load_data_fmnist(interest_class, args, is_to_q=True, is_shuffle=False,
                                              disable_visualize=False)
    else:
        raise Exception("The dataset is not supported!")

    # TODO(Note): Save the batch of images with amplitude encoding
    images_amp, labels_amp = next(iter(test_loader_amp))
    print("image shape: ", images_amp.shape)
    print(images_amp[0].shape)
    # torchvision.utils.save_image(images_amp[0], 'images_amp_{}.jpg'.format(img_size))

    # TODO(Note): Save the batch of original images with qubit extension
    # Prepare for the encoding circuits
    args.num_enc_qubits = 0
    num_enc_qubits = 0
    ori_args = Ori_encoding_args(device, args, pre_gate="hadamard", entag_pattern="identity")  # For baseline
    # print("max number of qubits is ", max_num_enc_qubits)
    if num_enc_qubits < max_num_enc_qubits:
        ori_enc_circuit = PSNR_Encoding_Circuit(num_ori_qubits, num_enc_qubits, max_num_enc_qubits, ori_args)
    else:
        ori_enc_circuit = Encoding_Circuit(ori_args)     # the same form with amp
    print("The model of the baseline comparison is as follows")
    print(ori_enc_circuit)

    images_ori_ext_fla = ori_enc_circuit(images_ori)  # [BS, 2 ** num_tol_qubits]
    images_ori_ext_fla = images_ori_ext_fla.to(torch.float32)
    images_ori_ext = images_ori_ext_fla.reshape(bsz, image_c, image_h, -1)
    print(images_ori_ext.shape)
    # m = nn.Upsample(scale_factor=(1, 1, 4, 1))
    # images_ori_ext = m(images_ori_ext)
    # torchvision.utils.save_image(images_ori_ext[0], 'images_no_amp_ext_{}.jpg'.format(img_size))

    # TODO(Note): Save the batch of images with amplitude encoding and qubit extension
    images_amp_ext_fla = ori_enc_circuit(images_amp)  # [BS, 2 ** num_tol_qubits]
    images_amp_ext_fla = images_amp_ext_fla.to(torch.float32)
    images_amp_ext = images_amp_ext_fla.reshape(bsz, image_c, image_h, -1)
    # torchvision.utils.save_image(images_amp_ext[0], 'images_amp_ext_{}.jpg'.format(img_size))

    # TODO(Note): Save the batch of images with amplitude encoding, qubit extension, 1-qubit encryption
    args.ry_angle_factor_list = ry_angle_list_set[0]
    print(args.ry_angle_factor_list)
    args.num_enc_qubits = 1
    num_enc_qubits = 1
    args.entag_pattern = 'single'
    # Prepare for the encoding circuits
    if num_enc_qubits < max_num_enc_qubits:
        enc_circuit_1 = PSNR_Encoding_Circuit(num_ori_qubits, num_enc_qubits, max_num_enc_qubits, args)
    else:
        enc_circuit_1 = Encoding_Circuit(args)
    print("The model of enc_circuit with 1-qubit encryption is as follows")
    print(enc_circuit_1)

    se1_images_amp_ext_fla = enc_circuit_1(images_amp)  # [BS, 2 ** num_tol_qubits]
    se1_images_amp_ext_fla = se1_images_amp_ext_fla.to(torch.float32)
    se1_images_amp_ext = se1_images_amp_ext_fla.reshape(bsz, image_c, image_h, -1)
    # torchvision.utils.save_image(se1_images_amp_ext[0], 'images_amp_ext_se1_{}.jpg'.format(img_size))

    # TODO(Note): Save the batch of images with amplitude encoding, qubit extension, 2-qubit encryption
    args.ry_angle_factor_list = ry_angle_list_set[1]
    print(args.ry_angle_factor_list)
    args.num_enc_qubits = 2
    num_enc_qubits = 2
    args.entag_pattern = 'single'

    # Prepare for the encoding circuits
    if num_enc_qubits < max_num_enc_qubits:
        enc_circuit_2 = PSNR_Encoding_Circuit(num_ori_qubits, num_enc_qubits, max_num_enc_qubits, args)
    else:
        enc_circuit_2 = Encoding_Circuit(args)
    print("The model of enc_circuit with 2-qubit encryption is as follows")
    print(enc_circuit_2)

    se2_images_amp_ext_fla = enc_circuit_2(images_amp)  # [BS, 2 ** num_tol_qubits]
    se2_images_amp_ext_fla = se2_images_amp_ext_fla.to(torch.float32)
    se2_images_amp_ext = se2_images_amp_ext_fla.reshape(bsz, image_c, image_h, -1)
    # torchvision.utils.save_image(se2_images_amp_ext[0], 'images_amp_ext_se2_{}.jpg'.format(img_size))

    # TODO(Note): calculate PSNR
    base_bsz_PSNR, base_avg_PSNR, base_bsz_mse, base_avg_mse = cal_PSNR(images_ori_ext_fla, images_amp_ext_fla)
    print("The PSNR for baseline is as follows")
    print(base_bsz_PSNR)

    q1_bsz_PSNR, q1_avg_PSNR, q1_bsz_mse, q1_avg_mse = cal_PSNR(images_ori_ext_fla, se1_images_amp_ext_fla)
    print("The PSNR for 1 qubit encryption is as follows")
    print(q1_bsz_PSNR)

    q2_bsz_PSNR, q2_avg_PSNR, q2_bsz_mse, q2_avg_mse = cal_PSNR(images_ori_ext_fla, se2_images_amp_ext_fla)
    print("The PSNR for 2 qubit encryption is as follows")
    print(q2_bsz_PSNR)

    index_dict = {i: -1 for i in range(10)}
    high_PNSR_dict = {i: -1 for i in range(10)}  # highest
    print(base_bsz_PSNR.shape[0])
    for index in range(base_bsz_PSNR.shape[0]):
        target = labels_ori[index].data.item()
        # print(target)
        if index_dict[target] == -1:
            index_dict[target] = index
            high_PNSR_dict[target] = base_bsz_PSNR[index].data.item()
        elif base_bsz_PSNR[index] > high_PNSR_dict[target]:
            index_dict[target] = index
            high_PNSR_dict[target] = base_bsz_PSNR[index].data.item()
        else:
            pass

    print(index_dict)
    print(high_PNSR_dict)

    # For baseline
    fig_num = 10
    columns = 2
    amp_images_before = []
    PSNR_before = []
    for target, index in index_dict.items():
        amp_images_before.append(images_amp_ext[index])
        PSNR_before.append(round(base_bsz_PSNR[index].data.item(), 2))

    amp_images = []
    PSNR = []
    offset = 5
    for j in range(0, 5):
        amp_images.append(amp_images_before[j])
        amp_images.append(amp_images_before[j + offset])
        PSNR.append(PSNR_before[j])
        PSNR.append(PSNR_before[j + offset])

    path = dataset + '_images_baseline_{}.jpg'.format(img_size)
    amp_images = torch.stack(amp_images)
    save_image(amp_images, fig_num, columns, PSNR, path)

    # path = 'images_baseline_1_{}.jpg'.format(img_size)
    # amp_images = amp_images[0:4]
    # save_image(amp_images, fig_num, columns, PSNR, path)
    #
    # amp_images = amp_images[4:9]
    # path = 'images_baseline_2_{}.jpg'.format(img_size)
    # save_image(amp_images, fig_num, columns, PSNR, path)

    # For 1 qubit
    PSNR_before = []
    se1_amp_images_before = []
    for target, index in index_dict.items():
        se1_amp_images_before.append(se1_images_amp_ext[index])
        PSNR_before.append(round(q1_bsz_PSNR[index].data.item(), 2))

    se1_amp_images = []
    PSNR = []
    se1_amp_images = []
    offset = 5
    for j in range(0, 5):
        se1_amp_images.append(se1_amp_images_before[j])
        se1_amp_images.append(se1_amp_images_before[j + offset])
        PSNR.append(PSNR_before[j])
        PSNR.append(PSNR_before[j + offset])

    path = dataset + '_images_qubit_1_{}.jpg'.format(img_size)
    se1_amp_images = torch.stack(se1_amp_images)
    save_image(se1_amp_images, fig_num, columns, PSNR, path)

    # For 2 qubit
    PSNR_before = []
    se2_amp_images_before = []
    for target, index in index_dict.items():
        se2_amp_images_before.append(se2_images_amp_ext[index])
        PSNR_before.append(round(q2_bsz_PSNR[index].data.item(), 2))

    se2_amp_images = []
    PSNR = []
    offset = 5
    for j in range(0, 5):
        se2_amp_images.append(se2_amp_images_before[j])
        se2_amp_images.append(se2_amp_images_before[j + offset])
        PSNR.append(PSNR_before[j])
        PSNR.append(PSNR_before[j + offset])

    path = dataset + '_images_qubit_2_{}.jpg'.format(img_size)
    se2_amp_images = torch.stack(se2_amp_images_before)
    save_image(se2_amp_images, fig_num, columns, PSNR, path)

    # For original
    Label_before = []
    original_images_before = []
    for target, index in index_dict.items():
        original_images_before.append(images_ori_ext[index])
        Label_before.append(target)

    Label = []
    original_images = []
    PSNR = []
    offset = 5
    for j in range(0, 5):
        original_images.append(original_images_before[j])
        original_images.append(original_images_before[j + offset])
        Label.append(Label_before[j])
        Label.append(Label_before[j + offset])

    path = dataset + '_images_original_{}.jpg'.format(img_size)
    original_amp_images = torch.stack(original_images)
    save_image_original(original_amp_images, fig_num, columns, Label, path)
