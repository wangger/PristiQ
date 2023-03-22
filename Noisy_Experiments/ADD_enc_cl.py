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
from Qiskit_Fast.S_Encode import Encoding_Circuit_Qiskit
from qiskit import Aer
from qiskit.compiler import transpile

logger = logging.getLogger(__name__)


def get_cir_len(qiskit_VQC, args):
    seed = args.seed
    optim_level = args.optim_level
    print("The optimization level is ", optim_level)
    backend_name = args.backend_name
    print("before transpile ", qiskit_VQC.depth())

    if backend_name == 'ideal_sim':
        # print("I am here")
        backend_sim = Aer.get_backend('qasm_simulator')  # 'statevector_simulator'
        qiskit_model_trans = transpile(qiskit_VQC, backend_sim, optimization_level=optim_level, seed_transpiler=seed)
        cir_len = qiskit_model_trans.depth()
        print("after transpile ", cir_len)
        print(qiskit_model_trans)
    else:  # real quantum backend
        backend = args.provider.get_backend(args.backend_name)
        qiskit_model_trans = transpile(qiskit_VQC, backend, optimization_level=optim_level, seed_transpiler=seed)
        cir_len = qiskit_model_trans.depth()
        print("after transpile ", cir_len)
        print(qiskit_model_trans)

    # qiskit_model_trans.draw(output='mpl', filename='circuit_after.jpg')
    return cir_len


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
                        level=logging.WARNING,  # logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    # TODO(NOTE): Load the account in advance
    from qiskit import IBMQ
    # # Me
    # API_token = "3eec804fe4d033aaf06c2566f8b4711c16fd802f10dde0075d1bf307340489ca5ea6045075c7c185272f2a9566a7a1630a" \
    #             "9cd2c60d52c6e6dbd901438ee2e4c5"
    # hub = 'ibm-q-education'  # 'ibm-q'
    # group = 'george-mason-uni-1'  # 'open'
    # project = 'hardware-acceler'  # 'main'
    # args.hub = hub
    # args.group = group
    # args.project = project

    # # MIT
    # API_token = '51a2a5d55d3e1d9683ab4f135fe6fbb84ecf3221765e19adb408699d43c6eaa238265059c3c2955ba59328634ffbd' \
    #             '88ba14d5386c947d22eb9a826e40811d626'
    # hub = 'ibm-q'
    # group = 'open'
    # project = 'main'
    # args.hub = hub
    # args.group = group
    # args.project = project

    # # Dr. Jiang
    # API_token = '782eee1e5e7bce7a91de590e309186db8f2386f0b588623aa414d128309510e62a4ee6820ffe650b8b7faf096c3ab684' \
    #             'e2b43048f279bf79c5df7dfa081554e6'
    # hub = 'ibm-q-education'  # 'ibm-q'
    # group = 'george-mason-uni-1'  # 'open'
    # project = 'hardware-acceler'  # 'main'
    # args.hub = hub
    # args.group = group
    # args.project = project

    # Junhuan
    API_token = '9eea23c32307ff886c5920319cbfe20ae3fa582737586a9f5a4ae125e301aa17c1312a3960f63e55b9b79b2021100f773' \
                '25e798ad8be3fa020620b3f49bfca80'
    hub = 'ibm-q'
    group = 'open'
    project = 'main'
    args.hub = hub
    args.group = group
    args.project = project

    # IBMQ.save_account(token=API_token, hub=hub, group=group, project=project, overwrite=True)
    # provider = IBMQ.load_account()  # Load account from disk
    # args.provider = provider

    provider = IBMQ.enable_account(token=API_token, hub=hub, group=group, project=project)
    args.provider = provider
    print(provider)
    print(IBMQ.stored_account())
    print(IBMQ.active_account())

    args.backend_name = 'ideal_sim' if args.backend_name is None else args.backend_name

    # handle permutation list:
    print("The permutation list before processing is ", args.permutation_list)
    if args.entag_pattern == 'random':  # if args.permutation_list != "None" or args.entag_pattern == 'random':
        permutation_list = args.permutation_list.strip('\'[').strip(']\'')
        permutation_list = permutation_list.replace(',', "")
        permutation_list = permutation_list.replace('\'', "")   # TODO: do we need this line?
        print(permutation_list)
        final_list = []
        for index in range(0, len(permutation_list), 2):
            pair = [int(permutation_list[index]), int(permutation_list[index + 1])]
            print("The pair is ", pair)
            final_list.append(pair)
        print(final_list)
        args.permutation_list = final_list

    print("The permutation list after processing is ", args.permutation_list)
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

    # TODO(NOTE): handle the important parameters for the baseline
    # Entag_pattern does not need to handle
    # num_enc_qubits does not need to handle
    # Permutation lists are handled in the block of building enc_circuit
    num_ori_qubits = args.num_ori_qubits
    num_enc_qubits = args.num_enc_qubits
    # total_qubits = num_ori_qubits + num_enc_qubits

    # handle ry_angle_list
    if args.ry_angle_factor_list == 'None':  # TODO(NOTE): New added
        args.pre_gate = 'identity'
        args.ry_angle_factor_list = None
    if args.ry_angle_factor_list is not None:
        ry_angle_factor_list = [float(x.strip()) for x in args.ry_angle_factor_list.split(",")]
        print(ry_angle_factor_list)
        args.ry_angle_factor_list = ry_angle_factor_list

    # # handle the permutation upper bound
    # fh_ub = num_ori_qubits * (num_ori_qubits - 1)
    # if args.fh_num_cnot is None:
    #     args.fh_num_cnot = fh_ub  # full pattern inside
    # elif args.fh_num_cnot >= fh_ub:
    #     args.fh_num_cnot = fh_ub
    #
    # lh_ub = num_ori_qubits * num_enc_qubits
    # if args.lh_num_cnot is None:
    #     args.lh_num_cnot = lh_ub  # full pattern inside
    # elif args.lh_num_cnot >= lh_ub:
    #     args.lh_num_cnot = lh_ub

    # TODO(Note): Load dataset
    dataset = args.dataset
    interest_class = [int(x.strip()) for x in args.interest_class.split(",")]
    n_ts_per_class = args.n_ts_per_class

    # if dataset == "mnist":
    #     # For mnist dataset
    #     # interest_class = args.interest_class
    #     args.output_num = len(interest_class)
    #     train_loader, test_loader = load_data_mnist(interest_class, args)  # used for final evaluation
    #
    # elif dataset == "mnist_general":
    #     interest_class = [int(x.strip()) for x in args.interest_class.split(",")]
    #     args.output_num = len(interest_class)
    #     output_num = args.output_num
    #     n_test_samples = n_ts_per_class * output_num
    #
    #     train_loader, test_loader = load_data_mnist_general(interest_class, args, is_to_q=True, is_shuffle=True,
    #                                                         disable_visualize=True, n_test_samples=n_test_samples)
    #
    # elif dataset == "fmnist_general":
    #     interest_class = [int(x.strip()) for x in args.interest_class.split(",")]
    #     args.output_num = len(interest_class)
    #     output_num = args.output_num
    #     n_test_samples = n_ts_per_class * output_num
    #
    #     train_loader, test_loader = load_data_fmnist_general(interest_class, args, is_to_q=True, is_shuffle=True,
    #                                                          disable_visualize=True, n_test_samples=n_test_samples)
    #
    # elif dataset == "fmnist":
    #     # For fashion mnist
    #     # interest_class = args.interest_class
    #     args.output_num = len(interest_class)
    #     train_loader, test_loader = load_data_fmnist(interest_class, args)  # used for final evaluation
    # else:
    #     raise Exception("The dataset is not supported!")

    # Build Encoding circuit
    Is_S_Encoding = args.Is_S_Encoding
    Is_ideal_acc = args.Is_ideal_acc
    if Is_S_Encoding:
        # TODO(Note): we use given permutation and angles
        if args.entag_pattern == 'random':
            pass
        else:
            args.permutation_list = None

        # prep_enc_circuit_generation(args)   # set args only once
        enc_circuit = Encoding_Circuit(args)  # Build encryption circuit in pytorch
        print("The model of enc_circuit is as follows")
        print(enc_circuit)

        # TODO(NOTE): Here we need to generate the circuit anyway
        qiskit_enc_circuit = build_enc_circuit(args)

        # if not Is_ideal_acc:
        #     # Build the encryption circuit in qiskit
        #     qiskit_enc_circuit = build_enc_circuit(args)
        #     # qiskit_enc_circuit = Encoding_Circuit_Qiskit(args)
        # else:
        #     qiskit_enc_circuit = None
    else:
        enc_circuit = None
        qiskit_enc_circuit = None

    print("The encoding circuit is as follows")
    print(qiskit_enc_circuit)

    # TODO(NOTE): Calculate the circuit length
    cir_len = get_cir_len(qiskit_enc_circuit, args)
    print(cir_len)

    # TODO(NOTE): Write the results back to the file
    backend_name = 'ideal_sim' if args.backend_name is None else args.backend_name
    args.backend_name = backend_name

    # assert args.Is_ideal_acc + args.Is_real_qc == 1, "error in setting the backend!"
    if args.Is_ideal_acc:
        output_dir = "Added_Output/Ideal/"
    elif args.Is_real_qc:
        output_dir = "Added_Output/Real_qc/"
    else:
        output_dir = "Added_Output/Noisy_sim/"

    out_filename = output_dir + dataset + "_" + args.interest_class + "_" + str(num_ori_qubits) + "_oriq_" \
                   + backend_name + "_pristiq_enc_cl.csv"

    record_dict = {}
    record_dict['num_enc_qubits'] = [args.num_enc_qubits]
    record_dict['ry_angle_factor_list'] = [args.ry_angle_factor_list]
    record_dict['entag_pattern'] = [args.entag_pattern]
    record_dict['permutation_list'] = [args.permutation_list]
    record_dict['enc_circuit_length'] = [cir_len]

    print(record_dict)
    record_dict_df = DataFrame(record_dict)

    if os.path.exists(out_filename):
        # print("file exists!")
        record_dict_df.to_csv(out_filename, mode='a+', index=False, header=False)
    else:
        # print("file created!")
        record_dict_df.to_csv(out_filename, mode='a+', index=False, header=True)






