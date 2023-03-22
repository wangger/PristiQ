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
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from NAS_train import test, train
import copy
from NAS_Evaluation import QNN_Evaluation_Noise_QC, QNN_Evaluation_Noise_Sim, QNN_Evaluation_Ideal


logger = logging.getLogger(__name__)


def Ideal_train(train_loader, valid_loader, model, logger, args):
    # TODO(NOTE): Decode the input parameters
    # data related
    device = args.device

    # training related
    init_lr = args.init_lr
    weight_decay = args.weight_decay
    max_epoch = args.max_epoch
    milestones = args.milestones
    # if args.milestones is not None:
    #     milestones = [int(x.strip()) for x in args.milestones.split(",")]   # for -m
    # else:
    #     milestones = [max_epoch * 0.5, max_epoch * 0.75]
    opt_type = args.optimizer
    sch_type = args.scheduler
    lr_decay_rate = args.lr_decay_rate

    # TODO (NOTE): Prepare for training
    criterion = nn.CrossEntropyLoss()   # TODO: Handle different choice of loss
    model = model.to(device)
    print("The model of VQC is as follows")
    print(model)

    if opt_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)
    else:
        raise Exception("The optimizer is not supported!")

    if sch_type == 'none':
        scheduler = None
    elif sch_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch)
    elif sch_type == 'linear':
        scheduler = MultiStepLR(optimizer, milestones, gamma=lr_decay_rate)
    else:
        raise Exception("The scheduler is not supported!")

    best_epoch, best_acc = 0, 0
    best_model = None
    epoch_init = 0
    loss_array = []

    init_acc, init_loss = test(model, criterion, train_loader, logger, args, enc_circuit=None)
    print("The initial accuracy on training set is ", init_acc)
    print("The initial loss on training set is ", init_loss)
    loss_array.append(init_loss)

    init_acc, init_loss = test(model, criterion, valid_loader, logger, args, enc_circuit=None)
    print("The initial accuracy on test set is ", init_acc)
    print("The initial loss on test set is ", init_loss)

    for epoch in range(epoch_init, max_epoch):
        print("=" * 20, epoch, "epoch", "=" * 20)
        print("Epoch Start at:", time.strftime("%m/%d/%Y %H:%M:%S"))

        print("-" * 20, "learning rates", "-" * 20)
        for param_group in optimizer.param_groups:
            print(param_group['lr'], end=",")  # we can set different lr for different part of params
        print()

        print("-" * 20, "training", "-" * 20)
        print("Training Start at:", time.strftime("%m/%d/%Y %H:%M:%S"))
        # TODO(Note): Train QNN
        loss = train(model, optimizer, epoch, criterion, train_loader, logger, args, enc_circuit=None)
        loss_array.append(loss)
        print("Training End at:", time.strftime("%m/%d/%Y %H:%M:%S"))
        print("-" * 60)

        print()

        print("-" * 20, "testing", "-" * 20)
        print("Testing Start at:", time.strftime("%m/%d/%Y %H:%M:%S"))
        cur_acc, _ = test(model, criterion, valid_loader, logger, args, enc_circuit=None)
        print("Testing End at:", time.strftime("%m/%d/%Y %H:%M:%S"))
        print("-" * 60)
        print()
        if scheduler is not None:
            scheduler.step()

        # is_best = False
        if cur_acc > best_acc:
            # is_best = True
            best_acc = cur_acc
            best_epoch = epoch
            del best_model  # Can it release some space?
            # TODO(Note): Should save the best model, but test set will become validation set
            best_model = copy.deepcopy(model)  # reference is not enough

        print("Best accuracy: {} at epoch {}; Current accuracy {}. Checkpointing".format(best_acc, best_epoch, cur_acc))
        print("Epoch End at:", time.strftime("%m/%d/%Y %H:%M:%S"))
        print("=" * 60)
        print()

    print("The final loss array is shown as follows.")
    print(loss_array)

    return best_acc, best_epoch, best_model

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
    n_ts_per_class = args.n_ts_per_class
    Is_ideal_acc = args.Is_ideal_acc    # TODO(Note): To specify whether to do ideal evaluation
    Is_real_qc = args.Is_real_qc

    # TODO(Note): Load dataset
    if dataset == "mnist":
        # For mnist dataset
        # interest_class = args.interest_class
        args.output_num = len(interest_class)
        train_loader, test_loader = load_data_mnist(interest_class, args)  # used for final evaluation

    elif dataset == "mnist_general":
        interest_class = [int(x.strip()) for x in args.interest_class.split(",")]
        args.output_num = len(interest_class)
        output_num = args.output_num
        n_test_samples = n_ts_per_class * output_num

        train_loader, test_loader = load_data_mnist_general(interest_class, args, is_to_q=True, is_shuffle=True,
                                                            disable_visualize=True, n_test_samples=n_test_samples)

    elif dataset == "fmnist_general":
        interest_class = [int(x.strip()) for x in args.interest_class.split(",")]
        args.output_num = len(interest_class)
        output_num = args.output_num
        n_test_samples = n_ts_per_class * output_num

        train_loader, test_loader = load_data_fmnist_general(interest_class, args, is_to_q=True, is_shuffle=True,
                                                             disable_visualize=True, n_test_samples=n_test_samples)
    elif dataset == "fmnist":
        # For fashion mnist
        # interest_class = args.interest_class
        args.output_num = len(interest_class)
        train_loader, test_loader = load_data_fmnist(interest_class, args)  # used for final evaluation
    else:
        raise Exception("The dataset is not supported!")

    # TODO(Note): Build the VQC
    layer_list = args.eval_arch
    layer_list = [layer for layer in layer_list.split(",")]
    # layer_list = ['v5', 'v19', 'v13', 'v2', 'v10', 'v0']

    # This is for pytorch model
    model = VQC_Net(layer_list, args)  # args.output_num is used
    for layer in model.q_layers:
        print(layer)

    # TODO(Note): Start to train the VQC ideally
    best_acc, best_epoch, best_model = Ideal_train(train_loader, test_loader, model, logger, args)

    # TODO(Note): Start to evaluate the best model on the test set
    test_acc = None

    if not Is_ideal_acc:
        if Is_real_qc:
            # evaluate the model on the ibmq machine
            test_acc = QNN_Evaluation_Noise_QC(train_loader, test_loader, best_model, None, args, logger,
                                               enc_circuit=None, qiskit_enc_circuit=None, Is_trained=True)
        else:
            # evaluate the model on the noisy simulator
            # train_loader, training_args are not used.
            test_acc = QNN_Evaluation_Noise_Sim(train_loader, test_loader, best_model, None, args, logger,
                                                enc_circuit=None, qiskit_enc_circuit=None, Is_trained=True)
    else:  # ideal case
        test_acc, _ = QNN_Evaluation_Ideal(train_loader, test_loader, best_model, None, args, logger,
                                           enc_circuit=None, Is_trained=True)

    # TODO(Note): calculate the number of parameters and circuit length
    n_paras = get_num_parameters(best_model)
    print("The number of parameters is ", n_paras)

    # This is for real computing qc
    update_layer_list = best_model.get_layer_list()  # It will filter v0
    layer_para_list = best_model.get_layer_para_list()
    # print(len(layer_para_list))
    # print(layer_para_list)
    qiskit_model = build_VQC_Net_Qiskit(update_layer_list, layer_para_list, args)

    # calculate the circuit length
    cir_len = get_cir_len(qiskit_model, args)   # args.optim_lvel, args.backend_name is used
    print(cir_len)

    # TODO(Note): Save the model
    backend_name = 'ideal_sim' if args.backend_name is None else args.backend_name
    args.backend_name = backend_name
    arch_str = ",".join(layer_list)

    if Is_ideal_acc:
        output_dir = "Added_Output/Basic_model/Ideal/"
    elif Is_real_qc:
        output_dir = "Added_Output/Basic_model/Real_qc/"
    else:
        output_dir = "Added_Output/Basic_model/Noisy_sim/"

    model_path = output_dir + dataset + "_" + args.interest_class + "_" + str(num_ori_qubits) + "_oriq_" \
                 + backend_name + "_arch_" + arch_str + "_basic_model.pth"
    torch.save(best_model, model_path)

    # TODO(NOTE): Write to the output file
    out_filename = output_dir + dataset + "_" + args.interest_class + "_" + str(num_ori_qubits) + "_oriq_" \
                   + backend_name + "_pristiq_basic_model.csv"

    record_dict = {}
    record_dict['NAS_best_acc_arch'] = [' '.join(layer_list)]
    record_dict['num_qubits'] = [args.num_ori_qubits + args.num_enc_qubits]
    record_dict['num_paras'] = [n_paras]
    record_dict['circuit_length'] = [cir_len]
    record_dict['test_acc'] = [test_acc]
    record_dict['basic_model_path'] = [model_path]

    record_dict_df = DataFrame(record_dict)

    if os.path.exists(out_filename):
        # print("file exists!")
        record_dict_df.to_csv(out_filename, mode='a+', index=False, header=False)
    else:
        # print("file created!")
        record_dict_df.to_csv(out_filename, mode='a+', index=False, header=True)

