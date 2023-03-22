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


class Ori_encoding_args:
    def __init__(self, device, args, pre_gate="identity", entag_pattern="identity",
                 ry_angle_factor_list=None):  # init method or constructor

        self.pre_gate = pre_gate
        self.entag_pattern = entag_pattern
        self.ry_angle_factor_list = ry_angle_factor_list
        self.device = device

        # TODO(Note): New added
        self.num_ori_qubits = args.num_ori_qubits
        self.num_enc_qubits = args.num_enc_qubits
        # self.Is_random_angles = args.Is_random_angles
        self.permutation_list = args.permutation_list


class PSNR_Encoding_Circuit(tq.QuantumModule):
    """
    Only useful when num_enc_qubits < max_num_enc_qubits
    forward:
        Input: original image x from dataloader. (bs, c, w, h)
        Output: encoded state vectors. (BS, 2**num_tol_qubits)
    """
    def __init__(self, num_ori_qubits, num_enc_qubits, max_num_enc_qubits, args):
        super().__init__()
        self.num_ori_qubits = num_ori_qubits
        self.num_enc_qubits = num_enc_qubits
        self.max_enc_qubits = max_num_enc_qubits    # TODO(Note): New added
        self.num_tol_qubits_clean = self.num_ori_qubits + self.num_enc_qubits   # TODO(Note): New added
        self.num_tol_qubits = self.num_ori_qubits + self.max_enc_qubits  # TODO(Note): New added

        self.start_loc = num_ori_qubits  # start index of added qubit
        self.pre_gate = args.pre_gate  # string
        self.entag_pattern = args.entag_pattern  # string
        if args.ry_angle_factor_list is not None:
            self.theta_list = [theta * math.pi for theta in args.ry_angle_factor_list]
        else:
            self.theta_list = None

        # TODO(Note): I added it
        self.permutation_list = args.permutation_list  # It could be None

        self.q_device = tq.QuantumDevice(n_wires=self.num_tol_qubits)
        self.device = args.device   # run on cuda/cpu
        # self.measure = tq.MeasureAll(tq.PauliZ) # Do not need it

    def _data_add_qubits(self, x):  # It will be called by forward first, handle a batch of data
        bsz = x.shape[0]    # The shape of inputs is torch.Size([BS, #C, #W, #H])
        original_states = x.view(bsz, -1, 1).to(C_DTYPE)  # TODO(Note): Change to the complex type
        original_states = original_states.to(self.device)  # [BS, # Amplitude, 1]

        single_added_states = torch.zeros(2 ** self.max_enc_qubits, dtype=C_DTYPE).to(self.device)   # [#Added_Amplitude]
        single_added_states[0] = 1 + 0j  # to make it the zero state
        repeat_times = [bsz] + [1] * len(single_added_states.shape)  # repeat for batch size -> [BS, #Added_Amplitude]
        batch_added_states = single_added_states.repeat(*repeat_times).to(self.device)
        batch_added_states = batch_added_states.view(bsz, -1, 1).to(self.device)    # [BS, #Added_Amplitude, 1]

        x = kronecker_product_einsum_batched(original_states, batch_added_states)   # [BS, ..., 1]
        x = torch.squeeze(x, dim=-1)    # [BS, ...]
        reshape_dim = [bsz] + [2] * self.num_tol_qubits
        x = torch.reshape(x, reshape_dim).to(self.device)

        return x    # The output is (BS, 2, 2, ...)

    def set_angle_list(self, angle_list):   # TODO(Note): New added for easily testing
        self.theta_list = [theta * math.pi for theta in angle_list]

    @tq.static_support
    def forward(self, x):

        # Add qubits to the batch of input data first
        x = self._data_add_qubits(x)    # (BS, 2, 2, ...)

        # Encode the data to self.states directly (add set method)
        self.q_device.set_states(x)

        # TODO(Note): Build circuit in the original qubits + enc qubits
        # Add some more non-parameterized gates (add on-the-fly) in wiki order!!!!
        # copy + scale
        if self.pre_gate == 'ry':
            for offset, theta in enumerate(self.theta_list):
                tqf.ry(self.q_device, wires=self.start_loc + offset, params=[theta], static=self.static_mode,
                       parent_graph=self.graph)
        elif self.pre_gate == 'hadamard':
            for index in range(self.start_loc, self.num_tol_qubits_clean):
                tqf.hadamard(self.q_device, wires=index, static=self.static_mode, parent_graph=self.graph)
        elif self.pre_gate == 'identity':
            # print("Pre_gate in id branch")
            pass  # No scaling operation for the copied input
        else:
            raise Exception("The gate is not supported by the encoding circuit!")

        # permutation part
        if self.entag_pattern == 'random':   # TODO(Note): I added it
            for cnot_wires in self.permutation_list:  # cannot be None
                tqf.cnot(self.q_device, wires=cnot_wires, static=self.static_mode, parent_graph=self.graph)

        if self.entag_pattern == 'single':
            # TODO (NOTE): The index setting is critical
            offset = self.num_ori_qubits    # used for target qubit loc
            for index in range(self.start_loc, self.num_tol_qubits_clean):
                # This is for a single CNOT
                index_c = index  # loc of the added qubit
                index_r = index_c - offset  # The original qubit is controlled by the added qubit
                tqf.cnot(self.q_device, wires=[index_c, index_r], static=self.static_mode, parent_graph=self.graph)

        elif self.entag_pattern == 'full':
            for c_index in range(self.start_loc, self.num_tol_qubits_clean):
                index_c = c_index  # loc of the added qubit
                for t_index in range(self.num_ori_qubits):  # entangle each added qubit with all of the original qubits
                    index_r = t_index  # Each original qubit is controlled by the added qubit
                    tqf.cnot(self.q_device, wires=[index_c, index_r], static=self.static_mode, parent_graph=self.graph)

        elif self.entag_pattern == 'identity':
            # print("Entanglement in id branch")
            pass  # No permutation operation for the copied input

        elif self.entag_pattern == 'single_add_0':
            for i in range(0, self.num_ori_qubits, 2):  # the depth is only 1
                if i > self.num_ori_qubits -2:
                    break
                else:
                    tqf.cnot(self.q_device, wires=[i, i+1], static=self.static_mode, parent_graph=self.graph)

            # TODO (NOTE): The index setting is critical
            offset = self.num_ori_qubits    # used for target qubit loc
            for index in range(self.start_loc, self.num_tol_qubits_clean):
                # This is for a single CNOT
                index_c = index  # loc of the added qubit
                index_r = index_c - offset  # The original qubit is controlled by the added qubit
                tqf.cnot(self.q_device, wires=[index_c, index_r], static=self.static_mode, parent_graph=self.graph)

        elif self.entag_pattern == 'single_add_1':
            for i in range(0, self.num_ori_qubits-1):  # the depth is only 1
                tqf.cnot(self.q_device, wires=[i, i + 1], static=self.static_mode, parent_graph=self.graph)

            # TODO (NOTE): The index setting is critical
            offset = self.num_ori_qubits  # used for target qubit loc
            for index in range(self.start_loc, self.num_tol_qubits_clean):
                # This is for a single CNOT
                index_c = index  # loc of the added qubit
                index_r = index_c - offset  # The original qubit is controlled by the added qubit
                tqf.cnot(self.q_device, wires=[index_c, index_r], static=self.static_mode, parent_graph=self.graph)
        else:
            raise Exception("This type of permutation is not currently supported!")

        # TODO (Note): New added, Add H gate for comparison
        for index in range(self.num_tol_qubits_clean, self.num_tol_qubits):
            tqf.hadamard(self.q_device, wires=index, static=self.static_mode, parent_graph=self.graph)

        # Return the encoded data
        x = self.q_device.get_states_1d().to(self.device)  # [BS, 2 ** num_tol_qubits]
        # print("The dtype inside encoding circuit is ", x.dtype)
        return x


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
    parser.add_argument('-dp', '--datapath', type=str, default='pytorch/data/', help='root path of the dataset of mnist/fmnist')

    # TODO(NOTE): Security encoding related, New added
    parser.add_argument("--Is_S_Encoding", help="Whether we want to do security encoding", action="store_true")
    parser.add_argument("--num_ori_qubits", type=int, default=4, help="Number of original qubits")
    parser.add_argument("--num_enc_qubits", type=int, default=0, help="Number of encoding qubits")

    # The following parameters are valid only when args.Is_S_Encoding is set
    parser.add_argument('--pre_gate', default='ry', choices=['identity', 'hadamard', 'ry'],
                        help='The gate for added qubits before entanglement (default=hadamard)')
    # parser.add_argument('--Is_random_angles', help="Whether we want to define the ry angles randomly",
    #                     action="store_true")
    parser.add_argument('--ry_angle_factor_list', nargs='+', type=float, default=None,
                        help='The list of angle factors (of pi) for added qubits')

    parser.add_argument('--entag_pattern', default='single', type=str, help='The pattern of entanglement for added '
                                                                            'qubits (default=single).')  # choices=['identity', 'single', 'full'],
    # parser.add_argument('-deb', "--debug", action="store_true", help="Debug mode")
    parser.add_argument("--permutation_list", type=str, default=None, help="given permutation list")

    # TODO(NOTE): Used for PSNR comparison, New added
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
    save_chkp = args.save_chkp  # for -chk

    # Set the random seed
    seed = args.seed
    fix_random_seeds(seed)

    # TODO(NOTE): Decode the input parameters
    # chkpath = args.chk_path
    has_cuda = torch.cuda.is_available()
    device = torch.device(args.device if has_cuda else "cpu")
    print("The program is running at {}".format(device))
    args.device = device    # to make device and args.device consistent, useful for the encoding

    # TODO(Note): Parameters related to security, Added
    Is_S_Encoding = args.Is_S_Encoding
    num_ori_qubits = args.num_ori_qubits
    num_enc_qubits = args.num_enc_qubits
    max_num_enc_qubits = args.max_num_enc_qubits
    args.pre_gate = 'ry'  # TODO(Note): Always ry
    # num_tol_qubits = num_ori_qubits + num_enc_qubits   # total circuit for the VQC!
    # pre_gate = 'ry'
    # entag_pattern = args.entag_pattern  # string

    print("=" * 21, "Your setting is listed as follows", "=" * 22)
    print("\t{:<25} {:<15}".format('Attribute', 'Input'))
    for k, v in vars(args).items():
        if v is not None:
            v = str(v)
            print("\t{:<25} {:<15}".format(k, v))
    print("=" * 22, "Exploration will start, have fun", "=" * 22)
    print("=" * 78)

    # TODO(Note): Calculate the PSNR of Baseline

    #  prepare the necessary encoding circuit for baseline calculation
    ori_args = Ori_encoding_args(device, args, pre_gate="hadamard", entag_pattern="identity")  # For baseline
    if num_enc_qubits < max_num_enc_qubits:
        ori_enc_circuit = PSNR_Encoding_Circuit(num_ori_qubits, num_enc_qubits, max_num_enc_qubits, ori_args)
    else:
        ori_enc_circuit = Encoding_Circuit(ori_args)   # the same form with amp, no qubit extension
    print("The model of the baseline comparison is as follows")
    print(ori_enc_circuit)


    # Initialize the security model
    if Is_S_Encoding:
        if num_enc_qubits < max_num_enc_qubits:
            enc_circuit = PSNR_Encoding_Circuit(num_ori_qubits, num_enc_qubits, max_num_enc_qubits, args)
        else:
            enc_circuit = Encoding_Circuit(args)
        print("The model of enc_circuit is as follows")
        print(enc_circuit)
    else:
        enc_circuit = None


    # data related
    interest_class = [int(x.strip()) for x in args.interest_class.split(",")]  # for -c
    args.preprocessdata = False  # TODO(Note): It should be always false
    # isppd = args.preprocessdata
    # datapath = args.datapath
    # img_size = args.img_size
    # num_workers = args.num_workers
    # batch_size = args.batch_size
    dataset = args.dataset
    interest_class = [int(x.strip()) for x in args.interest_class.split(",")]

    if dataset == "mnist":
        args.output_num = len(interest_class)
        # TODO(Note): Load data set w/ and w/o amplitude encoding, the shuffle of test dataset is false
        _, test_loader_clean = load_data_mnist(interest_class, args, is_to_q=False)
        _, test_loader_amp = load_data_mnist(interest_class, args, is_to_q=True)
    elif dataset == "fmnist":
        args.output_num = len(interest_class)
        _, test_loader_clean = load_data_fmnist(interest_class, args, is_to_q=False)
        _, test_loader_amp = load_data_fmnist(interest_class, args, is_to_q=True)
    else:
        raise Exception("The dataset is not supported!")

    # TODO(Note): To calculate the PSNR of the baseline
    base_total_PSNR = 0
    base_total_MSE = 0
    base_total_data_size = 0
    load_index = 0

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

    # # TODO(Note): Initialize the circuit to encode the images with identity method
    # # The comparison object is identity
    # ori_args = Ori_encoding_args(device)
    # ori_enc_circuit = Encoding_Circuit(num_ori_qubits, num_enc_qubits, ori_args)
    # The comparison object is identity
    # ori_args = Ori_encoding_args(device, pre_gate="hadamard", entag_pattern="identity")
    # ori_enc_circuit = Encoding_Circuit(num_ori_qubits, num_enc_qubits, ori_args)

    # TODO(Note): record the best result
    best_PSNR = 1000  # less is better
    best_MSE = 1000  # less is better
    best_angle_list = None  # not the set_list!

    # TODO(Note): for calculate the avg result
    avg_PSNR_list = []  # for each angle, there is an avg_PSNR
    avg_MSE_list = []

    set_angle_list = [[0.33, 0.67]]  # [[0.33], [0.67]]

    # For each sample of angles
    for ry_angle_factor_list in set_angle_list:  # TODO: Do not need the loop
        enc_circuit.set_angle_list(ry_angle_factor_list)  # list of float numbers
        print("The rotation angle of encoding for each ry gate ", ry_angle_factor_list)

        # Reset the variables
        total_PSNR = 0
        total_MSE = 0
        total_data_size = 0

        # iterate on test dataloader (1 epoch)
        # reset dataloader to correct the pointer
        # _, clean_test_loader = load_data(interest_class, args, is_to_q=False)
        # _, test_loader = load_data(interest_class, args, is_to_q=True)
        test_loader_clean_iter = iter(test_loader_clean)    # reset the iterator

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

            # print("The avg PSNR is {}".format(avg_PSNR))
            # print("The avg mse is {}".format(avg_mse))
            # print("The PSNR list is {}".format(bsz_PSNR))
            # print("The mse list is {}".format(bsz_mse))

        final_PSNR = total_PSNR / total_data_size
        final_MSE = total_MSE / total_data_size
        print("The rotation angles are {}".format(ry_angle_factor_list))
        print("The current final PSNR is {}".format(final_PSNR))
        print("The current final MSE is {}".format(final_MSE))
        avg_PSNR_list.append(final_PSNR)
        avg_MSE_list.append(final_MSE)

        if final_PSNR < best_PSNR:
            best_PSNR = final_PSNR
            best_MSE = final_MSE
            best_angle_list = ry_angle_factor_list

    print("The best PSNR is {}".format(best_PSNR))
    print("The best MSE is {}".format(best_MSE))
    print("The best angle list is {}".format(best_angle_list))

    # Calculate the average of metrics related to angle
    assert len(avg_PSNR_list) == len(set_angle_list) and len(avg_PSNR_list) == len(avg_MSE_list), "Wrong code!"
    avg_PSNR = sum(avg_PSNR_list)/len(avg_PSNR_list)
    avg_mse = sum(avg_MSE_list)/len(avg_MSE_list)
    print("The avg PSNR is {}".format(avg_PSNR))
    print("The avg MSE is {}".format(best_MSE))

    # # Extract two sub-figures
    # bs = se_amp_images.shape[0]  # TODO(NOTE): It must be divided by 2
    # image_c = se_amp_images.shape[1]  # number of channel
    # image_w = se_amp_images.shape[2]
    #
    # # se_amp_images = se_amp_images.reshape(bs, -1)
    # # print(se_amp_images[0].shape)
    # # print("Example after adding qubits", se_amp_images[0])
    #
    # se_amp_images = se_amp_images.reshape(bs, -1)
    # se_amp_images_0 = se_amp_images[0:bs:2]  # step is 2
    # se_amp_images_1 = se_amp_images[1:bs:2]  # step is 2
    # se_amp_images_0 = se_amp_images_0.reshape(bs, image_c, image_w, -1)
    # se_amp_images_1 = se_amp_images_1.reshape(bs, image_c, image_w, -1)
    #
    # print(se_amp_images_0.shape)
    # print(se_amp_images_1.shape)
    # torchvision.utils.save_image(se_amp_images_0, 'secure_images_encoding_0_{}.jpg'.format(img_size))
    # torchvision.utils.save_image(se_amp_images_1, 'secure_images_encoding_1_{}.jpg'.format(img_size))
