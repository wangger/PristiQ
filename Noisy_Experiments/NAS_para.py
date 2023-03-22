
import argparse
import sys
sys.path.append("../interfae/")


def parse_args():
    parser = argparse.ArgumentParser(description='QuantumFlow Classification Training')

    parser.add_argument('--seed', type=int, default=2, help='random seed')  # use cuda to specify GPU

    # Data related
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'custom_linear', 'fmnist', 'mnist_general', 'fmnist_general'],
                        help='The dataset used for this experiment')
    parser.add_argument('--device', type=str, default='cpu', help='device')  # use 'cuda' to specify GPU
    parser.add_argument('-c', '--interest_class', type=str, default="3, 6", help="investigate classes")  # for mnist/fmnist
    parser.add_argument('--img_size', type=int, default=4, help="row size for image down sampling")
    parser.add_argument('--img_size_col', type=int, default=None, help="column size for image down sampling")
    parser.add_argument('--n_ts_per_class', type=int, default=50, help="number of test samples per class")
    parser.add_argument('-j', '--num_workers', type=int, default=0, help="worker to load data", )
    parser.add_argument('-tb', '--batch_size', type=int, default=64, help="training batch size", )
    parser.add_argument('-ib', '--inference_batch_size', type=int, default=256, help="inference batch size", )
    parser.add_argument('-ppd', "--preprocessdata", action="store_true", help="Using the preprocessed data")  # only for mnist

    # For custom xor dataset, to get the filename of dataset
    parser.add_argument('--dim_features', type=int, default=8, help="dim of features")
    parser.add_argument('--num_train_samples', type=int, default=1000, help="number of training samples")
    parser.add_argument('--num_test_ratio', type=float, default=0.1, help="ratio of samples for testing "
                                                                          "related to train samples")

    # Pareto Training related
    parser.add_argument('-l', '--init_lr', type=float, default=0.05, help="PNN learning rate")
    parser.add_argument('-ld', '--lr_decay_rate', type=float, default=0.1, help="decay rate for lr")
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, help="weight decay")
    parser.add_argument('-opt', '--optimizer', default='adam', choices=['adam', 'sgd'],
                        help='The type of used optimizer')
    parser.add_argument('-sch', '--scheduler', default='none', choices=['none', 'cosine', 'linear'],
                        help='The type of scheduler')
    parser.add_argument('-m', '--milestones', type=str, default=None, help="Training milestone")  # default="3, 5, 8"
    parser.add_argument('-e', '--max_epoch', type=int, default=20, help="Training epoch")
    # parser.add_argument('-r', '--resume_path', type=str, default='', help='resume from checkpoint')
    # parser.add_argument('-t', '--test_only', help="Only Test without Training", action="store_true")
    # parser.add_argument('-bin', '--binary', help="binary activation", action="store_true")

    # NAS Training related
    parser.add_argument('--nas_init_lr', type=float, default=0.05, help="PNN learning rate")
    parser.add_argument('--nas_lr_decay_rate', type=float, default=0.1, help="decay rate for lr")
    parser.add_argument('--nas_weight_decay', type=float, default=0, help="weight decay")
    parser.add_argument('--nas_optimizer', default='adam', choices=['adam', 'sgd'],
                        help='The type of used optimizer')
    parser.add_argument('--nas_scheduler', default='none', choices=['none', 'cosine', 'linear'],
                        help='The type of scheduler')
    parser.add_argument('--nas_milestones', type=str, default=None, help="Training milestone")  # default="3, 5, 8"
    parser.add_argument('--nas_max_epoch', type=int, default=20, help="Training epoch")  #

    # # Model related
    # parser.add_argument('-nn', '--layers', type=str, default="v0:5", help="QNN structrue :<layer1 name: number of "
    #                     "this layer, layer2 name:number of this layer, ...")

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
    parser.add_argument('--Is_random_angles', help="Whether we want to define the ry angles randomly",
                        action="store_true")
    parser.add_argument('--random_angle_lb', type=float, default=0.125,
                        help="lower bound factor of pi for ry random rotation")
    parser.add_argument('--random_angle_ub', type=float, default=0.875,
                        help="upper bound factor of pi for ry random rotation")
    # The angle list should not be None if used
    # parser.add_argument('--ry_angle_factor_list', nargs='+', type=float, default=None,
    #                     help='The list of angle factors (of pi) for added qubits')
    # 0.33,0.67
    parser.add_argument('--ry_angle_factor_list', type=str, default=None, help='The list of angle factors (of pi) for '
                                                                               'added qubits')

    # choices=['identity', 'single', 'full', 'random', 'single_add_0', 'single_add_1']
    parser.add_argument('--entag_pattern', default='random', type=str, help='The pattern of entanglement for added '
                                                                            'qubits (default=random).')
    parser.add_argument('--fh_num_cnot', default=None, type=int, help='Number of random cnots in the first half part,'
                                                                      'i.e., the cnots within the original qubits')
    parser.add_argument('--lh_num_cnot', default=None, type=int, help='Number of random cnots in the last half part,'
                                                                      'i.e., the cnots between the added qubits and '
                                                                      'the original qubits')
    # parser.add_argument('-deb', "--debug", action="store_true", help="Debug mode")

    # TODO(NOTE): NAS related
    parser.add_argument("--Is_ideal_acc", help="In NAS, whether the validation for reward is on ideal simulators",
                        action="store_true")
    parser.add_argument('--reward_beta', type=float, default=1.0,
                        help="control the trade-off between transpiled circuit length and accuracy")
    parser.add_argument('--max_episodes', type=int, default=100, help="total number of episodes for NAS")

    # TODO(NOTE): pareto sampling related, always 1 in the experiments
    parser.add_argument('--num_child_QNN', default=1, type=int, help='Number of child QNN to sample for sampling')

    # TODO(NOTE): quantum backend related
    parser.add_argument('--backend_name', default=None, type=str, help='backend name for IBMQ execution/Noisy simulator')
    parser.add_argument('--num_shots', default=8092, type=int, help='Number of shots for IBMQ execution/Noisy simulator')
    parser.add_argument('--optim_level', default=3, type=int, help='optimization level for IBMQ execution/Noisy '
                                                                   'simulator')
    # TODO(NOTE): Baseline related
    parser.add_argument("--Not_Expanded", action="store_true", help="whether expanded the original model")
    # e.g., ['2,1','2,3','2,1','0,2','0,3','2,3','5,0','5,0','4,3','4,0','4,3','4,0','4,2','5,2'], str
    parser.add_argument("--permutation_list", type=str, default=None, help="given permutation list")
    # e.g., v2,v0,v2,v5,v5
    parser.add_argument("--eval_arch", type=str, default=None, help="given model architecture")

    # TODO(NOTE): security related
    parser.add_argument("--basic_path", type=str, default=None, help="the path of the basic model")
    parser.add_argument("--Is_real_qc", action="store_true", help="whether to evaluate it on real qc")
    parser.add_argument("--basic_dir", type=str, default=None, help="the parent dir of the basic model")

    args = parser.parse_args()
    return args


# import torchquantum as tq
# import torchquantum.functional as tqf
# import torch.nn.functional as F
#
#
# class QFCModel(tq.QuantumModule):
#     class QLayer(tq.QuantumModule):
#         def __init__(self):
#             super().__init__()
#             self.n_wires = 4
#             self.random_layer = tq.RandomLayer(n_ops=50,
#                                                wires=list(range(self.n_wires)))
#
#             # gates with trainable parameters
#             self.rx0 = tq.RX(has_params=True, trainable=True)
#             self.ry0 = tq.RY(has_params=True, trainable=True)
#             self.rz0 = tq.RZ(has_params=True, trainable=True)
#             self.crx0 = tq.CRX(has_params=True, trainable=True)
#
#         @tq.static_support
#         def forward(self, q_device: tq.QuantumDevice):
#             """
#             1. To convert tq QuantumModule to qiskit or run in the static
#             model, need to:
#                 (1) add @tq.static_support before the forward
#                 (2) make sure to add
#                     static=self.static_mode and
#                     parent_graph=self.graph
#                     to all the tqf functions, such as tqf.hadamard below
#             """
#             self.q_device = q_device
#
#             self.random_layer(self.q_device)
#
#             # some trainable gates (instantiated ahead of time)
#             self.rx0(self.q_device, wires=0)
#             self.ry0(self.q_device, wires=1)
#             self.rz0(self.q_device, wires=3)
#             self.crx0(self.q_device, wires=[0, 2])
#
#             # add some more non-parameterized gates (add on-the-fly)
#             tqf.hadamard(self.q_device, wires=3, static=self.static_mode,
#                          parent_graph=self.graph)
#             tqf.sx(self.q_device, wires=2, static=self.static_mode,
#                    parent_graph=self.graph)
#             tqf.cnot(self.q_device, wires=[3, 0], static=self.static_mode,
#                      parent_graph=self.graph)
#
#     def __init__(self):
#         super().__init__()
#         self.n_wires = 4
#         self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
#         self.encoder = tq.GeneralEncoder(
#             tq.encoder_op_list_name_dict['4x4_ryzxy'])
#
#         self.q_layer = self.QLayer()
#         self.measure = tq.MeasureAll(tq.PauliZ)
#
#     def forward(self, x, use_qiskit=False):
#         bsz = x.shape[0]
#         x = F.avg_pool2d(x, 6).view(bsz, 16)
#
#         if use_qiskit:
#             x = self.qiskit_processor.process_parameterized(
#                 self.q_device, self.encoder, self.q_layer, self.measure, x)
#         else:
#             self.encoder(self.q_device, x)
#             self.q_layer(self.q_device)
#             x = self.measure(self.q_device)
#
#         x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
#         x = F.log_softmax(x, dim=1)
#
#         return x
#
#     def get_para(self):
#         for name, parameters in self.q_layer.rx0.named_parameters():
#             print(name, ':', parameters.size())
#             print(name, ':', parameters.data.item())
#             print(name, ':', type(parameters.data.item()))
#
#
# class training_args():
#     def __init__(self):
#         self.Is_built = True
#
#
# if __name__ == "__main__":
#     model = QFCModel()
#     model.get_para()
#
#     print(type(model.named_parameters()))
#     print(model)
#     for name, parameters in model.q_layer.rx0.named_parameters():
#         print(name, ':', parameters.size())
#         print(name, ':', parameters.data.item())
#         print(name, ':', type(parameters.data.item()))
#
#     print(type(model.named_parameters()))
#
#     import torch
#     import time
#     test = time.time()
#     print(test)
#     filename = 'test_model_sdsdsdhjsdhsjdshdjsdhsjdhsjdshdjshdjsdhsjdshdjsshjsd' + str(test) + '.pth'
#     torch.save(model, filename)
#
#     new_model = torch.load(filename)
#
#     print(new_model)
#     for name, parameters in new_model.q_layer.rx0.named_parameters():
#         print(name, ':', parameters.size())
#         print(name, ':', parameters.data.item())
#         print(name, ':', type(parameters.data.item()))


# test = training_args()
# test.lr = 0.01
# test.decay = 0.02
# print(test.lr)
# print(test.decay)


if __name__ == "__main__":
    # import pandas as pd
    # df = pd.DataFrame(columns=['lib', 'qty1', 'qty2'])
    # row = {}
    # row['lib'] = [['v0', 'v2', 'v3']]  # it should be 2-dim
    # row['qty1'] = 0.023
    # row['qty2'] = 2
    # # df = df.append(row, ignore_index=True)
    # row_df = pd.DataFrame(row)
    # print(row_df)
    # df = pd.concat([df, row_df], ignore_index=True)
    # print(df)
    #
    # row = {}
    # row['lib'] = [['v0', 'v2', 'v3']]  # it should be 2-dim
    # row['qty1'] = 0.023
    # row['qty2'] = 2
    # # df = df.append(row, ignore_index=True)
    # row_df = pd.DataFrame(row)
    # print(row_df)
    # df = pd.concat([df, row_df], ignore_index=True)
    # print(df)

    # from c_input import load_data_fmnist, load_data_mnist
    # import torch
    # args = parse_args()
    # interest_class = [3, 4, 5]
    # train_loader, test_loader = load_data_fmnist(interest_class, args)
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     print(data.shape)
    #     print(data[0])
    #     print(target[0])
    #     bsz = data.shape[0]
    #     data = data.reshape(bsz, -1)
    #     norm = torch.norm(data, dim=1)
    #     print(norm.shape)
    #     print(norm)
    #     break
    # print("done")

    # args = parse_args()

    # def modify_args(args):
    #     args.called = True
    #     args.nb = 'hhhhhh'
    #
    # modify_args(args)
    #
    # print(args.called)
    # print(args.nb)
    #
    # def modify_args_2(args):
    #     print(args.nb)
    #     args.nb = 'yeah!'
    #
    # modify_args_2(args)
    #
    # print(args.nb)
    # print(args.called)
    #
    # import time
    # test = time.time()
    # print(test)

    # args = parse_args()
    # args_dict = {}
    # for k, v in vars(args).items():
    #     if v is not None:
    #         v = str(v)
    #     else:
    #         v = "None"
    #     print("\t{:<25} {:<15}".format(k, v))
    #     args_dict[k] = v
    # dict_df = None

    pass
