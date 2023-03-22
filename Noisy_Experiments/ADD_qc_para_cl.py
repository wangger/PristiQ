import torch
from c_input import *
import logging
from NAS_train import Training_Ideal
from NAS_para import parse_args
from utils import fix_random_seeds
from S_Encode import prep_enc_circuit_generation, Encoding_Circuit
from S_Encode_qiskit import build_enc_circuit
from utils import get_cir_len_baseline
from pandas import DataFrame
import time
from Qiskit_Fast.S_Encode import Encoding_Circuit_Qiskit
import sys
import random
from NAS_Net import VQC_Net
from NAS_Net_Qiskit import build_VQC_Net_Qiskit
from qiskit import Aer
from qiskit.compiler import transpile
logger = logging.getLogger(__name__)


def get_num_parameters(model):
    """
    calculate the learnable parameters
    """
    n_para = 0
    for layer in model.q_layers:
        for name, param in layer.named_parameters():
            if param.requires_grad:
                n_para += 1

                # print(name, param)
                # print(param.shape)
                # print(param.requires_grad)
    return n_para


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
    else:  # real quantum backend
        backend = args.provider.get_backend(args.backend_name)
        qiskit_model_trans = transpile(qiskit_VQC, backend, optimization_level=optim_level, seed_transpiler=seed)
        cir_len = qiskit_model_trans.depth()
        print("after transpile ", cir_len)

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
    num_ori_qubits = args.num_ori_qubits
    num_enc_qubits = args.num_enc_qubits
    # total_qubits = num_ori_qubits + num_enc_qubits

    dataset = args.dataset
    interest_class = [int(x.strip()) for x in args.interest_class.split(",")]
    args.output_num = len(interest_class)

    # Build Encoding circuit
    # Is_S_Encoding = args.Is_S_Encoding
    # Is_ideal_acc = args.Is_ideal_acc

    # TODO(Note): Build the VQC
    layer_list = args.eval_arch
    layer_list = [layer for layer in layer_list.split(",")]
    # layer_list = ['v5', 'v19', 'v13', 'v2', 'v10', 'v0']

    # This is for pytorch model
    torch_VQC = VQC_Net(layer_list, args)  # args.output_num is used
    for layer in torch_VQC.q_layers:
        print(layer)

    # TODO(Note): calculate the number of parameters
    n_paras = get_num_parameters(torch_VQC)
    print("The number of parameters is ", n_paras)

    # This is for real computing qc
    update_layer_list = torch_VQC.get_layer_list()  # It will filter v0
    layer_para_list = torch_VQC.get_layer_para_list()
    # print(len(layer_para_list))
    # print(layer_para_list)
    qiskit_model = build_VQC_Net_Qiskit(update_layer_list, layer_para_list, args)

    # calculate the circuit length
    cir_len = get_cir_len(qiskit_model, args)   # args.optim_lvel, args.backend_name is used
    print(cir_len)

    # TODO(NOTE): Write to the output file
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
                   + backend_name + "_pristiq_qc_para_cl.csv"

    record_dict = {}
    record_dict['NAS_best_acc_arch'] = [' '.join(layer_list)]
    record_dict['num_enc_qubits'] = [num_enc_qubits]
    record_dict['num_paras'] = [n_paras]
    record_dict['circuit_length'] = [cir_len]
    record_dict_df = DataFrame(record_dict)

    if os.path.exists(out_filename):
        # print("file exists!")
        record_dict_df.to_csv(out_filename, mode='a+', index=False, header=False)
    else:
        # print("file created!")
        record_dict_df.to_csv(out_filename, mode='a+', index=False, header=True)

