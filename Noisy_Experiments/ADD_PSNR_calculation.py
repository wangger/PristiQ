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
sys.path.append("../interface/")
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
from PSNR_func import *  # TODO(NOTE): This function is written by myself
import pandas as pd
from pandas import DataFrame

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
        # print("max number of qubits is inside", self.max_enc_qubits)

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
        # print("reshape_dim is", reshape_dim)
        # print("total qubits", self.num_tol_qubits)
        # print("x shape after expanded is", x.shape)
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
        # print("Inside the PSNR_enc function,", type(self.entag_pattern), self.entag_pattern)

        if self.entag_pattern == 'random':   # TODO(Note): I added it
            for cnot_wires in self.permutation_list:  # cannot be None
                tqf.cnot(self.q_device, wires=cnot_wires, static=self.static_mode, parent_graph=self.graph)
        elif self.entag_pattern == 'single':
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
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'fmnist', 'mnist_general',
                                                               'fmnist_general'],
                        help='The dataset used for this experiment')
    parser.add_argument('--device', type=str, default='cpu', help='device')  # use 'cuda' to specify GPU
    parser.add_argument('-c', '--interest_class', type=str, default="3, 6", help="investigate classes")
    parser.add_argument('-s', '--img_size', type=int, default=4, help="image size 4: 4*4")
    parser.add_argument('--img_size_col', type=int, default=None, help="column size for image down sampling")
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
    # parser.add_argument("--num_enc_qubits", type=int, default=0, help="Number of encoding qubits")

    # The following parameters are valid only when args.Is_S_Encoding is set
    # parser.add_argument('--pre_gate', default='ry', choices=['identity', 'hadamard', 'ry'],
    #                     help='The gate for added qubits before entanglement (default=hadamard)')
    # parser.add_argument('--Is_random_angles', help="Whether we want to define the ry angles randomly",
    #                     action="store_true")
    # parser.add_argument('--ry_angle_factor_list', nargs='+', type=float, default=None,
    #                     help='The list of angle factors (of pi) for added qubits')
    # parser.add_argument('--ry_angle_factor_list', type=str, default=None, help='The list of angle factors (of pi) for '
    #                                                                            'added qubits')

    # parser.add_argument('--entag_pattern', default='single', type=str, help='The pattern of entanglement for added '
    #                                                                         'qubits (default=single).')  # choices=['identity', 'single', 'full'],
    # parser.add_argument('-deb', "--debug", action="store_true", help="Debug mode")
    # parser.add_argument("--permutation_list", type=str, default=None, help="given permutation list")

    # TODO(NOTE): Used for PSNR comparison, New added
    parser.add_argument("--max_num_enc_qubits", type=int, default=0, help="Number of maximum encoding qubits for "
                                                                          "comparison")
    # parser.add_argument('--max_episodes', type=int, default=100, help="total number of episodes for NAS")
    parser.add_argument('--backend_name', default=None, type=str,
                        help='backend name for IBMQ execution/Noisy simulator')
    parser.add_argument('--n_ts_per_class', type=int, default=50, help="number of test samples per class")
    parser.add_argument("--Is_ideal_acc", help="In NAS, whether the validation for reward is on ideal simulators",
                        action="store_true")
    parser.add_argument("--Is_real_qc", action="store_true", help="whether to evaluate it on real qc")


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
    fix_random_seeds(seed)  # TODO(NOTE): Important for test trainset sampling

    # TODO(NOTE): Decode the input parameters
    # chkpath = args.chk_path
    has_cuda = torch.cuda.is_available()
    device = torch.device(args.device if has_cuda else "cpu")
    print("The program is running at {}".format(device))
    args.device = device    # to make device and args.device consistent, useful for the encoding

    # TODO(Note): Parameters related to security, Added
    Is_S_Encoding = args.Is_S_Encoding
    num_ori_qubits = args.num_ori_qubits
    # num_enc_qubits = args.num_enc_qubits
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

    # load the dataset for PSNR caculation
    # interest_class = [int(x.strip()) for x in args.interest_class.split(",")]  # for -c
    args.preprocessdata = False  # TODO(Note): It should be always false
    # isppd = args.preprocessdata
    # datapath = args.datapath
    # img_size = args.img_size
    # num_workers = args.num_workers
    # batch_size = args.batch_size
    dataset = args.dataset
    interest_class = [int(x.strip()) for x in args.interest_class.split(",")]
    n_ts_per_class = args.n_ts_per_class

    if dataset == "mnist":
        args.output_num = len(interest_class)
        # TODO(Note): Load data set w/ and w/o amplitude encoding, the shuffle of test dataset is false
        _, test_loader_clean = load_data_mnist(interest_class, args, is_to_q=False)
        _, test_loader_amp = load_data_mnist(interest_class, args, is_to_q=True)
    elif dataset == "fmnist":
        args.output_num = len(interest_class)
        _, test_loader_clean = load_data_fmnist(interest_class, args, is_to_q=False)
        _, test_loader_amp = load_data_fmnist(interest_class, args, is_to_q=True)

    elif dataset == "mnist_general":
        args.output_num = len(interest_class)
        output_num = args.output_num
        n_test_samples = n_ts_per_class * output_num

        _, test_loader_clean = load_data_mnist_general(interest_class, args, is_to_q=False, disable_visualize=True,
                                                       n_test_samples=n_test_samples)
        torch.manual_seed(seed)  # TODO(NOTE)
        _, test_loader_amp = load_data_mnist_general(interest_class, args, is_to_q=True, disable_visualize=True,
                                                     n_test_samples=n_test_samples)

    elif dataset == "fmnist_general":
        args.output_num = len(interest_class)
        output_num = args.output_num
        n_test_samples = n_ts_per_class * output_num
        _, test_loader_clean = load_data_fmnist_general(interest_class, args, is_to_q=False, disable_visualize=True,
                                                        n_test_samples=n_test_samples)
        torch.manual_seed(seed)  # TODO(NOTE)
        _, test_loader_amp = load_data_fmnist_general(interest_class, args, is_to_q=True, disable_visualize=True,
                                                      n_test_samples=n_test_samples)
    else:
        raise Exception("The dataset is not supported!")

    # TODO(Note): Calculate the PSNR of Baseline
    # Prepare the necessary encoding circuit for baseline calculation
    args.num_enc_qubits = 0
    args.permutation_list = None
    num_enc_qubits = args.num_enc_qubits  # always 0 for baseline TODO: Maybe should not add it?
    ori_args = Ori_encoding_args(device, args, pre_gate="hadamard", entag_pattern="identity")  # For baseline
    if num_enc_qubits < max_num_enc_qubits:
        ori_enc_circuit = PSNR_Encoding_Circuit(num_ori_qubits, num_enc_qubits, max_num_enc_qubits, ori_args)
    else:
        ori_enc_circuit = Encoding_Circuit(ori_args)   # the same form with amp, no qubit extension
    print("The model of the baseline comparison is as follows")
    print(ori_enc_circuit)

    base_final_PSNR, base_final_MSE = PSNR_Baseline_cal(test_loader_clean, test_loader_amp,
                                                        ori_enc_circuit, args)  # only args.device will be used

    # TODO(Note): Start the loop to get the PSNR of each key for a given dataset
    # read the information from the input file
    # Input_filename = "Baseline_Readin/" + dataset + "_" + args.interest_class + "_baseline_input.csv"
    # backend_name = args.backend_name
    # max_episodes = args.max_episodes
    # Input_filename = "Baseline_Readin/" + dataset + "_" + args.interest_class + "_" + str(args.num_ori_qubits) \
    #                  + "_oriq_" + str(backend_name) + "_episode_" + str(max_episodes) + ".csv"

    backend_name = 'ideal_sim' if args.backend_name is None else args.backend_name
    args.backend_name = backend_name
    Is_ideal_acc = args.Is_ideal_acc
    Is_real_qc = args.Is_real_qc
    if Is_ideal_acc:
        input_dir = "Added_Readin/Ideal/"
    elif Is_real_qc:
        input_dir = "Added_Readin/Real_qc/"
    else:
        input_dir = "Added_Readin/Noisy_sim/"

    Input_filename = input_dir + dataset + "_" + args.interest_class + "_" + str(num_ori_qubits) + "_oriq_" + backend_name + \
                     "_pristiq_input.csv"

    df = pd.read_csv(Input_filename)
    list_num_enc_qubits = []
    list_ry_angle_factor_list = []
    list_entag_pattern = []
    list_permutation_list = []  # TODO(Note): It could be 'None' when entag_pattern is not random

    list_Avg_PSNR = []
    list_MSE = []

    list_skip_indices = []   # TODO(NOTE): Special handle of the baseline

    row_idx = 0
    for row in df["ry_angle_factor_list"]:
        if row == 'None':  # TODO(NOTE): Special handle of the baseline
            list_skip_indices.append(row_idx)
            row_idx += 1
            continue
        row = row.strip("[").strip("]").replace(" ", "")
        # convert to float list
        ry_angle_factor_list = [float(x.strip()) for x in row.split(",")]
        list_ry_angle_factor_list.append(ry_angle_factor_list)
        row_idx += 1

    row_idx = 0
    for row in df["num_enc_qubits"]:  # int
        if row_idx in list_skip_indices:    # TODO(NOTE): Special handle of the baseline
            row_idx += 1
            continue
        if not isinstance(row, int):
            row = int(row)
        list_num_enc_qubits.append(row)
        row_idx += 1

    row_idx = 0
    for row in df["entag_pattern"]:
        if row_idx in list_skip_indices:    # TODO(NOTE): Special handle of the baseline
            row_idx += 1
            continue
        list_entag_pattern.append(row)
        row_idx += 1

    row_idx = 0
    for row in df["permutation_list"]:
        if row_idx in list_skip_indices:    # TODO(NOTE): Special handle of the baseline
            row_idx += 1
            continue
        if row != "None":
            row = [x.strip(' [').strip('[').strip(']').replace(" ", "") for x in row.split("],")]
            row = str(row)
            row = row.replace(" ", "")
            # convert to int list of list
            row = [[int(y.strip("\'")) for y in x.split(",")] for x in row.strip('[').strip(']').split('\',\'')]

        list_permutation_list.append(row)
        row_idx += 1

    # start the loop
    for index in range(len(list_num_enc_qubits)):
        # reset the necessary parameters
        args.num_enc_qubits = list_num_enc_qubits[index]
        args.ry_angle_factor_list = list_ry_angle_factor_list[index]
        args.entag_pattern = list_entag_pattern[index]
        args.permutation_list = list_permutation_list[index]

        print("=========================")
        print(args.num_enc_qubits)
        print(args.ry_angle_factor_list)
        print(args.entag_pattern)
        print(args.permutation_list)
        print("=========================")

        # Initialize the security circuit
        if Is_S_Encoding:
            if num_enc_qubits < max_num_enc_qubits:
                enc_circuit = PSNR_Encoding_Circuit(args.num_ori_qubits, args.num_enc_qubits, args.max_num_enc_qubits,
                                                    args)  # not ori_args
            else:
                enc_circuit = Encoding_Circuit(args)
            print("The model of enc_circuit is as follows")
            print(enc_circuit)
        else:
            enc_circuit = None

        final_PSNR, final_MSE = PSNR_Key_cal(test_loader_clean, test_loader_amp, ori_enc_circuit, enc_circuit, args)

        # TODO(Note): record the output
        list_Avg_PSNR.append(float(final_PSNR))
        list_MSE.append(final_MSE)

    # TODO(Note): Write to output file
    # Result_filename = "PSNR_Result/" + args.dataset + "_" + args.interest_class + "_maxq_" + str(args.max_num_enc_qubits) \
    #                   + "_PSNR.csv"

    # Result_filename = "PSNR_Result/" + args.dataset + "_" + args.interest_class + "_" + str(args.num_ori_qubits) \
    #                   + "_oriq_" + str(args.max_num_enc_qubits) + "_maxq_" + str(backend_name) + "_episode_" \
    #                   + str(max_episodes) + "_PSNR.csv"

    backend_name = 'ideal_sim' if args.backend_name is None else args.backend_name
    args.backend_name = backend_name
    # assert args.Is_ideal_acc + args.Is_real_qc == 1, "error in setting the backend!"
    if args.Is_ideal_acc:
        output_dir = "Added_Output/Ideal/"
    elif args.Is_real_qc:
        output_dir = "Added_Output/Real_qc/"
    else:
        output_dir = "Added_Output/Noisy_sim/"

    Result_filename = output_dir + dataset + "_" + args.interest_class + "_" + str(num_ori_qubits) + "_oriq_" \
                      + backend_name + "_PSNR.csv"

    output_dict = {}
    output_dict["num_enc_qubits"] = list_num_enc_qubits
    output_dict["ry_angle_factor_list"] = list_ry_angle_factor_list
    output_dict["entag_pattern"] = list_entag_pattern
    output_dict["permutation_list"] = list_permutation_list
    output_dict["Avg_PSNR"] = list_Avg_PSNR

    # We could write header here into
    output_dict_df = DataFrame(output_dict)
    output_dict_df.to_csv(Result_filename, mode='a+', index=False)

    # write the baseline result
    baseline_dict = {}
    baseline_dict["num_enc_qubits"] = [0]
    baseline_dict["ry_angle_factor_list"] = ['None']
    baseline_dict["entag_pattern"] = ['identity']
    baseline_dict["permutation_list"] = ['None']
    baseline_dict["Avg_PSNR"] = [float(base_final_PSNR)]  # tensor -> float

    baseline_dict_df = DataFrame(baseline_dict)
    baseline_dict_df.to_csv(Result_filename, mode='a+', index=False, header=None)






