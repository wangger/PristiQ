import torch
from c_input import *
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.nn as nn

import torchquantum as tq
import torchquantum.functional as tqf

from NAS_dict import layer_name_dict
from NAS_Net import VQC_Net
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from NAS_train import Training_Ideal
from NAS_para import parse_args
from utils import fix_random_seeds
from RL_NAS.rl_controller import *
from S_Encode import prep_enc_circuit_generation, Encoding_Circuit
from S_Encode_qiskit import build_enc_circuit
from custom_dataset_generation import load_data_custom_xor
from utils import get_cir_len_baseline
from pandas import DataFrame
import time

# TODO: Check the import packages. Use the NAS packages
# TODO(Note): currently, we encrypt the data on the fly for each batch (not offline)
# TODO: train set? valid set and test set?
# TODO: How to train tensorflow in cuda
# TODO(NOTE): In the current version of torchquantum, I changed the devices.py. maybe also measure.py?

logger = logging.getLogger(__name__)


class NAS_training_args():
    def __init__(self):
        self.Is_built = True


class pareto_training_args():   # for the final retraining of sampled QNN from trained RNN controller
    def __init__(self):
        self.Is_built = True


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    print("=" * 21, "Your setting is listed as follows", "=" * 22)
    print("\t{:<25} {:<15}".format('Attribute', 'Input'))
    for k, v in vars(args).items():
        if v is not None:
            v = str(v)
        else:
            v = "None"
        print("\t{:<25} {:<15}".format(k, v))

    print("=" * 22, "Exploration will start, have fun", "=" * 22)
    print("=" * 78)

    # get the logger for following usage
    logging.basicConfig(stream=sys.stdout,
                        level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    # TODO(Note): Set the random seed
    # fix_random_seeds(args.seed)
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # set seed for torchvision.transforms
    random.seed(seed)
    # np_seed = np.random.randint(10000)

    # set the device to run the pytorch part
    has_cuda = torch.cuda.is_available()
    device = torch.device(args.device if has_cuda else "cpu")
    print("The program is running at {}".format(device))

    if args.device == "cuda":   # acceleration for training
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    args.device = device  # to make device and args.device consistent, useful for the encoding

    # TODO(Note): handle ry_angle_list
    # ry_angle_factor_list = [x.strip() for x in args.ry_angle_factor_list.split(",")]
    # print(ry_angle_factor_list)
    if args.ry_angle_factor_list is not None:
        ry_angle_factor_list = [float(x.strip()) for x in args.ry_angle_factor_list.split(",")]
        print(ry_angle_factor_list)
        args.ry_angle_factor_list = ry_angle_factor_list

    # handle the permutation upper bound
    num_ori_qubits = args.num_ori_qubits
    num_enc_qubits = args.num_enc_qubits
    # total_qubits = num_ori_qubits + num_enc_qubits

    fh_ub = num_ori_qubits * (num_ori_qubits - 1)
    if args.fh_num_cnot is None:
        args.fh_num_cnot = fh_ub  # full pattern inside
    elif args.fh_num_cnot >= fh_ub:
        args.fh_num_cnot = fh_ub

    lh_ub = num_ori_qubits * num_enc_qubits
    if args.lh_num_cnot is None:
        args.lh_num_cnot = lh_ub  # full pattern inside
    elif args.lh_num_cnot >= lh_ub:
        args.lh_num_cnot = lh_ub

    # TODO(Note): Load dataset
    dataset = args.dataset
    interest_class = [int(x.strip()) for x in args.interest_class.split(",")]

    if dataset == "mnist":
        # For mnist dataset
        # interest_class = args.interest_class
        args.output_num = len(interest_class)
        train_loader, test_loader = load_data_mnist(interest_class, args)  # used for final evaluation
    elif dataset == "custom_xor":
        args.output_num = 2
        dim_features = args.dim_features
        num_train_samples = args.num_train_samples
        num_test_samples = int(num_train_samples * args.num_test_ratio)

        train_filename = "custom_xor_trainset_dim{}_size{}.csv".format(dim_features, num_train_samples)
        test_filename = "custom_xor_testset_dim{}_size{}.csv".format(dim_features, num_test_samples)
        train_loader, test_loader = load_data_custom_xor(train_filename, test_filename, args)
    elif dataset == "fmnist":
        # For fashion mnist
        # interest_class = args.interest_class
        args.output_num = len(interest_class)
        train_loader, test_loader = load_data_fmnist(interest_class, args)  # used for final evaluation
    else:
        raise Exception("The dataset is not supported!")

    # Split train_loader to train_loader and valid_loader for NAS next time
    NAS_train_loader = train_loader
    NAS_valid_loader = test_loader

    # TODO(Note): To generate the encryption circuit
    Is_S_Encoding = args.Is_S_Encoding
    Is_ideal_acc = args.Is_ideal_acc
    enc_circuit = None
    qiskit_enc_circuit = None
    args.permutation_list = None

    if Is_S_Encoding:
        prep_enc_circuit_generation(args)   # set arges only once

        enc_circuit = Encoding_Circuit(args)  # Build encryption circuit in pytorch
        print("The model of enc_circuit is as follows")
        print(enc_circuit)
        if not Is_ideal_acc:
            # Build the encryption circuit in qiskit
            qiskit_enc_circuit = build_enc_circuit(args)

    # TODO(Note): Get the training setting for NAS
    nas_args = NAS_training_args()
    nas_args.init_lr = args.nas_init_lr
    nas_args.weight_decay = args.nas_weight_decay
    nas_args.max_epoch = args.nas_max_epoch

    if args.nas_milestones is not None:
        nas_args.milestones = [int(x.strip()) for x in args.nas_milestones.split(",")]  # for -m
    else:
        nas_args.milestones = [nas_args.max_epoch * 0.5, nas_args.max_epoch * 0.75]

    nas_args.optimizer = args.nas_optimizer
    nas_args.scheduler = args.nas_scheduler
    nas_args.lr_decay_rate = args.nas_lr_decay_rate

    # TODO(Note): Get the training setting for final sampling
    pareto_args = pareto_training_args()
    pareto_args.init_lr = args.init_lr
    pareto_args.weight_decay = args.weight_decay
    pareto_args.max_epoch = args.max_epoch

    if args.milestones is not None:
        pareto_args.milestones = [int(x.strip()) for x in args.milestones.split(",")]  # for -m
    else:
        pareto_args.milestones = [pareto_args.max_epoch * 0.5, pareto_args.max_epoch * 0.75]

    pareto_args.optimizer = args.optimizer
    pareto_args.scheduler = args.scheduler
    pareto_args.lr_decay_rate = args.lr_decay_rate

    # TODO(Note): Get the file flag
    file_flag = str(time.time()) + str(np.random.randint(10000))
    print("The file flag is ", file_flag)
    args.file_flag = file_flag

    # TODO(NOTE): Run the search algorithm and get the pareto front
    print("-"*20 + "Start to search for the best QNN" + "-"*20)
    # get the baseline for reward calculation first
    args.cir_baseline = get_cir_len_baseline(args)  # only used for ideal simulator
    print(args.cir_baseline)
    print("get the cir_baseline")

    controller = Controller(logger)   # Initialize
    # train the RNN controller
    # TODO: Might need to adapt
    controller.global_train(NAS_train_loader, NAS_valid_loader, enc_circuit, qiskit_enc_circuit, nas_args, args)
    # sample from the trained RNN controller and get the pareto front
    print("-"*20 + "start to sample the last QNN" + "-"*20)
    # TODO: Might need to adapt
    controller.get_pareto_front(train_loader, test_loader, enc_circuit, qiskit_enc_circuit, pareto_args, args)
    print("-"*20 + "Finish to search for the best QNN" + "-"*20)
    print("The permutation list is ", args.permutation_list)
    print("The entangle pattern is", args.entag_pattern)
    print("The angle rotation list is ", args.ry_angle_factor_list)

    # TODO(NOTE): Write the episode result
    controller.write_results_to_csv(args)

    log_filename = "Experimental_Result/" + str(args.dataset) + "_" + str(args.interest_class) + "_beta_" + \
                   str(args.reward_beta) + "_encq_" + str(args.num_enc_qubits) + "_" + str(args.file_flag) + ".csv"
    # TODO(Note): Write the highlighted information
    highlight_dict = {}
    highlight_dict['file_flag'] = args.file_flag

    # information about dataset
    highlight_dict['dataset'] = args.dataset
    highlight_dict['interest_class'] = args.interest_class

    # information about NAS and training
    highlight_dict['reward_beta'] = args.reward_beta
    highlight_dict['nas_max_epoch'] = args.nas_max_epoch
    highlight_dict['max_epoch'] = args.max_epoch

    # key information
    highlight_dict['Is_S_Encoding'] = args.Is_S_Encoding
    highlight_dict['num_enc_qubits'] = args.num_enc_qubits
    highlight_dict['Is_random_angles'] = args.Is_random_angles
    highlight_dict['ry_angle_factor_list'] = args.ry_angle_factor_list
    highlight_dict['entag_pattern'] = args.entag_pattern
    highlight_dict['permutation_list'] = args.permutation_list

    # make them list, clear none
    for k, v in highlight_dict.items():
        if v is not None:
            # v = str(v)
            pass
        else:
            v = "None"
        print("\t{:<25} {:<15}".format(k, str(v)))
        if not isinstance(v, list):
            v = [v]
        highlight_dict[k] = v

    print(highlight_dict)
    highlight_dict_df = DataFrame(highlight_dict)
    print(highlight_dict_df)
    highlight_dict_df.to_csv(log_filename, mode='a+', index=False)

    # TODO(Note): Write all the parameters at the end
    print("-"*20 + "write the args at the end" + "-"*20)
    args_dict = {}
    for k, v in vars(args).items():
        if v is not None:
            # v = str(v)
            pass
        else:
            v = "None"
        print("\t{:<25} {:<15}".format(k, str(v)))
        if v is not isinstance(v, list):
            v = [v]
        args_dict[k] = v

    print(args_dict)
    args_dict_df = DataFrame(args_dict)
    print(args_dict_df)
    args_dict_df.to_csv(log_filename, mode='a+', index=False)

    end_time = time.time()
    tol_time = round((end_time - start_time)/60, 3)
    # print(start_time)
    # print(end_time)
    # tol_time = end_time - start_time/60
    print("The file flag is ", args.file_flag)
    print("The total execution time is {} minutes".format(tol_time))



