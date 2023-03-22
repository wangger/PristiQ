# %%

import logging
import csv
import numpy as np
# import tensorflow as tf  # TODO: Is this version correct?
# import tensorflow
import tensorflow.compat.v1 as tf
import sys
from RL_NAS.rl_input import controller_params  # , HW_constraints
# import termplotlib as tpl
import copy
import random
from datetime import datetime
import time
import torch
import os
from NAS_Net import VQC_Net
from NAS_Evaluation import QNN_Evaluation_Ideal, QNN_Evaluation_Noise_QC, QNN_Evaluation_Noise_Sim
# import tensorflow_addons as tfa
from pandas import DataFrame
import matplotlib.pyplot as plt

# logger = logging.getLogger(__name__)  # TODO: Usage?


def ema(values):
    """
    Helper function for keeping track of an exponential moving average of a list of values.
    For this module, we use it to maintain an exponential moving average of rewards
    """
    weights = np.exp(np.linspace(-1., 0., len(values)))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode="full")[:len(values)]
    return a[-1]


class Controller(object):

    def __init__(self, logger):
        self.logger = logger
        self.graph = tf.Graph()

        # config = tf.ConfigProto()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=self.graph)  # It is used for executing the feedforward and backward
        # self.sess = tf.compat.v1.Session(config=config, graph=self.graph)  # It is used for executing the feedforward and backward
        self.hidden_units = controller_params['hidden_units']

        self.nn1_search_space = controller_params['sw_space']
        self.hw1_search_space = controller_params['hw_space']   # [] # TODO: hardware

        self.nn1_num_para = len(self.nn1_search_space)
        self.hw1_num_para = len(self.hw1_search_space)  # len = 0 # TODO: hardware

        # n_time_step
        self.num_para = self.nn1_num_para + self.hw1_num_para   # TODO: hardware

        self.nn1_beg, self.nn1_end = 0, self.nn1_num_para
        # self.hw1_beg, self.hw1_end = self.nn1_end, self.nn1_end + self.hw1_num_para  # TODO: hardware

        # TODO(Note): {key: index of time-step, value: list of options, not the indices of options}
        self.para_2_val = {}
        idx = 0
        for hp in self.nn1_search_space:    # for each layer
            self.para_2_val[idx] = hp
            idx += 1
        for hp in self.hw1_search_space:    # never execute # TODO: hardware
            self.para_2_val[idx] = hp
            idx += 1

        # key size = n_time_step, for each value -> (BS, n_options_i) after classifier for each time step w/o softmax
        self.RNN_classifier = {}
        # key size = n_time_step, for each value -> (BS, n_options_i) after classifier for each time step w/ softmax
        self.RNN_pred_prob = {}

        # TODO(NOTE): Build the compute graph, including the forward and the optimizer
        with self.graph.as_default():
            self.build_controller()

        self.reward_history = []    # record the mean reward of all the episode
        self.QNN_history = []   # record all the QNNs in all the episodes, with names # TODO: BS=1
        self.valid_acc_history = []  # record all the circuit length in all the episodes, with names # TODO: BS=1
        self.circuit_length_history = []    # record all the circuit length in all the episodes, with names # TODO: BS=1

        self.architecture_history = []  # record the batch of sampled child NN for all the episode, with index
        self.trained_network = {}   # save the performance of trained child NN
        # self.explored_info = {}     # the same function with trained_network in our case

        # self.target_HW_Eff = HW_constraints["target_HW_Eff"]    # TODO: hardware

        self.pareto_input = None

    def build_controller(self):
        self.logger.info('Building RNN Network')

        # Build inputs and placeholders
        with tf.name_scope('controller_inputs'):

            # TODO(NOTE): Wired, it has two usages!
            # [1] Input to the NASCell, the scalar for each time-step will be embedded to a vector with n_hidden values
            # [2] The sampled actions for all the time steps
            # [BS, n_time_step]
            # for usage[1], it will be embedded first -> [BS, n_time_step, n_hidden]


            self.child_network_paras = tf.placeholder(tf.int64, [None, self.num_para], name='controller_input')
            # self.child_network_paras = tf.compat.v1.placeholder(tf.int64, [None, self.num_para], name='controller_input')

            # Discounted rewards, [BS, ], need to get the reward for each input sample within batch for gradient cal
            # discounted means we need [absolute reward - ema reward]
            self.discounted_rewards = tf.placeholder(tf.float32, (None,), name='discounted_rewards')
            # self.discounted_rewards = tf.compat.v1.placeholder(tf.float32, (None,), name='discounted_rewards')

            # batch size for RNN, have to be 1 in this code
            self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')   # scalar
            # self.batch_size = tf.compat.v1.placeholder(tf.int32, [], name='batch_size')  # scalar

        # Build the embedding dict and convert the original input to the embedded vector
        # embedding dict -> self.embedding_weights, embedded vector -> self.embedded_input
        with tf.name_scope('embedding'):
            self.embedding_weights = []  # list with len = n_time_step, element i is (n_options_i, n_hidden)

            # Build the embedding dict
            embedding_id = 0
            para_2_emb_id = {}  # {key = time_step_i, value =time_step_i}
            for i in range(len(self.para_2_val.keys())):    # for each time step
                additional_para_size = len(self.para_2_val[i])  # num of options for each layer
                additional_para_weights = tf.get_variable('state_embeddings_%d' % (embedding_id),
                                                          shape=[additional_para_size, self.hidden_units],
                                                          initializer=tf.initializers.random_uniform(-1., 1.))
                # additional_para_weights = tf.compat.v1.get_variable('state_embeddings_%d' % (embedding_id),
                #                                                     shape=[additional_para_size, self.hidden_units],
                #                                                     initializer=tf.initializers.random_uniform(-1., 1.))
                self.embedding_weights.append(additional_para_weights)
                para_2_emb_id[i] = embedding_id  # embedding
                embedding_id += 1

            # Build the embedded vector
            self.embedded_input_list = []  # list with len = n_time_step, (BS, n_hidden)
            for i in range(self.num_para):  # for each time step
                self.embedded_input_list.append(
                    tf.nn.embedding_lookup(self.embedding_weights[para_2_emb_id[i]], self.child_network_paras[:, i]))

            self.embedded_input = tf.stack(self.embedded_input_list, axis=-1)  # (BS, num_hidden, n_time_step)
            self.embedded_input = tf.transpose(self.embedded_input, perm=[0, 2, 1])  # (BS, n_time_step, n_hidden)

        self.logger.info('Building Controller')
        with tf.name_scope('controller'):
            with tf.variable_scope('RNN'):
            # with tf.compat.v1.variable_scope('RNN'):
                # nas = tf.contrib.rnn.NASCell(self.hidden_units)
                # nas = tfa.rnn.NASCell(self.hidden_units)
                # tmp_state = nas.zero_state(batch_size=self.batch_size, dtype=tf.float32)
                nas = tf.nn.rnn_cell.LSTMCell(self.hidden_units)
                tmp_state = nas.zero_state(batch_size=self.batch_size, dtype=tf.float32)
                init_state = tf.nn.rnn_cell.LSTMStateTuple(tmp_state[0], tmp_state[1])  # (2, BS, n_hidden)

                # output is (BS, n_time_step, n_hidden), final_state (2, BS, n_hidden) -> len of tuple is 2
                output, final_state = tf.nn.dynamic_rnn(nas, self.embedded_input, initial_state=init_state,
                                                        dtype=tf.float32)

                # TODO(NOTE): Build classifier for all the time steps
                # tmp_list = []   # [(BS), .., (BS)], most likely option list for all the time steps TODO: useless
                # print("output","="*50,output)
                # print("output slice","="*50,output[:,-1,:])
                for para_idx in range(self.num_para):   # for each time step
                    o = output[:, para_idx, :]  # (bs, n_hidden)
                    para_len = len(self.para_2_val[para_idx])   # n_options_i
                    # len(self.para_val[para_idx % self.para_per_layer])
                    classifier = tf.layers.dense(o, units=para_len, name='classifier_%d' % (para_idx), reuse=False)
                    self.RNN_classifier[para_idx] = classifier  # (BS, n_options_i)
                    prob_pred = tf.nn.softmax(classifier)   # (BS, n_options_i)
                    self.RNN_pred_prob[para_idx] = prob_pred
                    # child_para = tf.argmax(prob_pred, axis=-1)  # (bs),the option with maximum prob TODO: useless
                    # tmp_list.append(child_para)
                # self.pred_val = tf.stack(tmp_list, axis=1)  # (BS, n_time_step), TODO: useless

        self.logger.info('Building Optimization')
        # Global Optimization composes all RNNs in one, like NAS, where arch_idx = 0
        with tf.name_scope('Optimizer'):
            self.global_step = tf.Variable(0, trainable=False)  # TODO: ?
            self.learning_rate = tf.train.exponential_decay(0.99, self.global_step, 50, 0.5, staircase=True)
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

        # We separately compute loss of prob dist for each time step since the dim of each prob dist may not be same
        with tf.name_scope('Loss'):
            for para_idx in range(self.num_para):  # for each time step
                if para_idx == 0:
                    # TODO(NOTE): The second usage of self.child_network_paras
                    # i.e., The sampled actions for all the time steps
                    self.policy_gradient_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.RNN_classifier[para_idx], labels=self.child_network_paras[:, para_idx])
                else:
                    self.policy_gradient_loss = tf.add(self.policy_gradient_loss,
                                                       tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                           logits=self.RNN_classifier[para_idx],
                                                           labels=self.child_network_paras[:, para_idx]))
            # get mean of loss
            self.policy_gradient_loss /= self.num_para  # average on time
            self.total_loss = self.policy_gradient_loss  # (BS)
            # TODO: ? shape is (num_gradients, BS)
            self.gradients = self.optimizer.compute_gradients(self.total_loss)  # policy gradients (BS, num_gradients)

            # TODO: ? average on batch for each grad of parameter
            for i, (grad, var) in enumerate(self.gradients):  # updated with policy gradients
                if grad is not None:  # for each grad   # TODO: ? what is var
                    # print("aaa",grad)
                    # print("aaa",self.discounted_rewards)
                    # sys.exit(0)

                    self.gradients[i] = (grad * self.discounted_rewards, var)   # multiply with reward for each sample

        with tf.name_scope('Train_RNN'):
            # The main training operation. This applies REINFORCE on the weights of the Controller
            # self.train_operation[arch_idx][pip_idx] = self.optimizer[arch_idx][pip_idx].apply_gradients(self.gradients[arch_idx][pip_idx], global_step=self.global_step[arch_idx][pip_idx])
            # self.train_operation = self.optimizer.minimize(self.total_loss)
            self.train_operation = self.optimizer.apply_gradients(self.gradients)   # update weights
            # TODO: ? usage
            self.update_global_step = tf.assign(self.global_step, self.global_step + 1, name='update_global_step')

        self.logger.info('Successfully built controller')

    def child_network_translate(self, child_network):   # DNA -> [[0, 3, 2, ...]], index, not the key value of option dict
        dnn_out = np.zeros_like(child_network)  # (1, n_time_step)   # TODO: BS=1
        dnn_out = dnn_out.tolist()
        print(child_network[0])
        print(self.para_2_val)
        for para_idx in range(self.num_para):   # for each time step
            # self.para_2_val[para_idx] is a list of options
            # print(para_idx)
            # print(child_network[0][para_idx])
            # print(self.para_2_val[para_idx])  #
            # print(dnn_out[0])
            # print(self.para_2_val[para_idx][child_network[0][para_idx]])
            # print(type(self.para_2_val[para_idx][child_network[0][para_idx]]))
            dnn_out[0][para_idx] = self.para_2_val[para_idx][child_network[0][para_idx]]  # TODO: BS=1
        print(dnn_out[0])
        return dnn_out  # [["v1", "v2", "v10", ....]] <- [[1, 2, 10, ...]] if index 1 -> "v1", ...

    def generate_child_network(self, child_network_architecture):
        with self.graph.as_default():
            feed_dict = {
                self.child_network_paras: child_network_architecture,
                self.batch_size: 1  # TODO: BS=1
            }
            # [(bs, n_options_i), ..., ], length is n_time_step
            rnn_out = self.sess.run(self.RNN_pred_prob, feed_dict=feed_dict)    # TODO(NOTE): Run the RNN
            predict_child = np.array([[0] * self.num_para])  # TODO: BS=1
            # random.seed(datetime.now())
            for para_idx, prob in rnn_out.items():  # for each time step
                predict_child[0][para_idx] = np.random.choice(range(len(self.para_2_val[para_idx])), p=prob[0])  # TODO: BS=1
            hyperparameters = self.child_network_translate(predict_child)  # TODO: BS=1, predict_child[0] is the generated DNA
            return predict_child, hyperparameters   # DNA for dict index, the real useful DNA for get the name of neuron

    def plot_history(self, history, filename, ylim=(-1, 1), title="reward"):  # TODO: save fig?
        x = list(range(len(history)))
        y = history
        # fig = tpl.figure()
        # fig.plot(x, y, ylim=ylim, width=60, height=20, title=title)
        # fig.show()

        plt.plot(x, y, 'b--')
        plt.title(title)
        plt.ylim(ylim[0], ylim[1])
        plt.savefig(filename, dpi=300)

    # def get_HW_efficiency(self, Network, HW1, RC):
    #     # Weiwen 01-24: Using the built Network and HW1 explored results to generate hardware efficiency
    #     # with the consideration of resource constraint RC
    #     return random.uniform(0, 1)

    def para2interface_NN(self, Para_NN1, args):
        # Weiwen 01-24: Build NN using explored hyperparamters, return Network
        Network = VQC_Net(Para_NN1, args)   # func(Para_NN1)
        return Network

    # def para2interface_HW(self, Para_HW1):
    #     # Weiwen 01-24: Build hardware model using the explored paramters
    #     HW1 = -1        # func(Para_HW1)
    #     RC = [HW_constraints["r_Ports_BW"],
    #           HW_constraints["r_DSP"],
    #           HW_constraints["r_BRAM"],
    #           HW_constraints["r_BRAM_Size"],
    #           HW_constraints["BITWIDTH"]]
    #     return HW1, RC

    # TODO: The interface might be changed for the different training settings
    def global_train(self, train_loader, valid_loader, enc_circuit, qiskit_enc_circuit, nas_args, args):
        logger = self.logger
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
        step = 0
        # total_rewards = 0   # TODO: useless

        # First input to the RNN
        child_network = np.array([[0] * self.num_para], dtype=np.int64)  # TODO: BS=1
        max_episodes = args.max_episodes
        # plt_filename = "Experimental_Result/reward_" + str(args.dataset) + "_" + str(args.interest_class) + "_" + str(
        # args.file_flag) + ".jpg"
        plt_filename = "Experimental_Result/reward_" + str(args.dataset) + "_" + str(
            args.interest_class) + "_beta_" + str(args.reward_beta) + "_encq_" + str(args.num_enc_qubits) + "_" + str(
            args.file_flag) + ".jpg"

        for episode in range(max_episodes):  # range(controller_params['max_episodes']):
            logger.info(
                '=-=-==-=-==-=-==-=-==-=-==-=-==-=-=>Episode {}<=-=-==-=-==-=-==-=-==-=-==-=-==-=-='.format(episode))
            step += 1
            episode_reward_buffer = []  # (BS, ), record the reward within the single episode/batch
            architecture_batch = []    # (BS, ), record the sampled DNA (indices) within the single episode/batch

            if episode % 50 == 0 and episode != 0:
                print("Process:", str(float(episode) / max_episodes * 100) + "%", file=sys.stderr)
                self.plot_history(self.reward_history, plt_filename, ylim=(min(self.reward_history)-0.01,
                                                                           max(self.reward_history) + 0.01))

            for sub_child in range(controller_params["num_children_per_episode"]):  # for each sample, sub_child is a index
                # TODO(NOTE): 1 + 2 + 3. feed child_network(input), Sample a child network from RNN
                # TODO(NOTE): 1. Generate the input (sampled child_network) to the RNN for sampling
                #  the next child network
                # child_network is the indices to the options in time steps, fit for being the input for next sample
                # hyperparameters is the values in the option dict, suitable to be the DNA to generate
                child_network, hyperparameters = self.generate_child_network(child_network)  # TODO: BS=1

                DNA_NN1 = child_network[0][self.nn1_beg:self.nn1_end]   # TODO: BS=1
                # DNA_HW1 = child_network[0][self.hw1_beg:self.hw1_end]   # TODO: hardware # TODO: BS=1

                Para_NN1 = hyperparameters[0][self.nn1_beg:self.nn1_end]    # TODO: BS=1
                # Para_HW1 = hyperparameters[0][self.hw1_beg:self.hw1_end]    # TODO: hardware # TODO: BS=1

                str_NN1 = " ".join(str(x) for x in Para_NN1)  # used for index and print
                str_NNs = str_NN1   # used for index and print

                # str_HW1 = " ".join(str(x) for x in Para_HW1)    # used for index and print   # TODO: hardware
                # str_HWs = str_HW1   # used for index and print   # TODO: hardware

                logger.info('=====>Step {}/{} in episode {}: HyperParameters: {} <====='.format(sub_child,
                                                                                                controller_params
                                                                                                ["num_children"
                                                                                                 "_per_episode"],
                                                                                                episode,
                                                                                                hyperparameters))
                # TODO(Note): Evaluate the sampled child network
                # if str_NNs in self.explored_info.keys():
                if str_NNs in self.trained_network.keys():  # if the NN has been explored, get the reward directly
                    # accuracy = self.explored_info[str_NNs][0]
                    # reward = self.explored_info[str_NNs][1]
                    if args.Is_ideal_acc:
                        valid_acc = self.trained_network[str_NNs]['acc']
                        cir_len = self.trained_network[str_NNs]['cir_len']  # we have this key
                        reward = self.trained_network[str_NNs]['reward']
                    else:
                        valid_acc = self.trained_network[str_NNs]['acc']    # acc on noisy device
                        reward = self.trained_network[str_NNs]['reward']

                    # HW_Eff = self.explored_info[str_NNs][2]

                else:   # if the NN has not been trained yet, evaluate them here
                    Network = self.para2interface_NN(Para_NN1, args)
                    if args.Is_ideal_acc:  # evaluate on ideal simulator
                        # we need both the accuracy and the circuit length in this case
                        # Here we train network and obtain the metrics for updating controller
                        # TODO: The interface might be changed for the different training settings
                        valid_acc, cir_len = QNN_Evaluation_Ideal(train_loader, valid_loader, Network, nas_args, args,
                                                                  logger, enc_circuit=enc_circuit)  # valid_acc < 1
                        # TODO(Note): Calculate the reward
                        beta = args.reward_beta
                        cir_baseline = args.cir_baseline
                        reward = valid_acc - beta * (cir_len/cir_baseline)

                        # Keep history trained data
                        self.trained_network[str_NNs] = {}
                        self.trained_network[str_NNs]['acc'] = valid_acc
                        self.trained_network[str_NNs]['cir_len'] = cir_len
                        self.trained_network[str_NNs]['reward'] = reward
                    else:
                        # TODO(Note): validate the ideally trained model obtained with best epoch
                        #  on the noisy device/simulator
                        # TODO: The interface might be changed for the different training settings
                        # valid_acc = QNN_Evaluation_Noise_QC(train_loader, valid_loader, Network, nas_args, args, logger,
                        #                                     enc_circuit=enc_circuit,
                        #                                     qiskit_enc_circuit=qiskit_enc_circuit)
                        valid_acc = QNN_Evaluation_Noise_Sim(train_loader, valid_loader, Network, nas_args, args, logger,
                                                             enc_circuit=enc_circuit,
                                                             qiskit_enc_circuit=qiskit_enc_circuit)
                        # TODO(Note): Calculate the reward
                        reward = valid_acc
                        # Keep history trained data
                        self.trained_network[str_NNs]['acc'] = valid_acc    # acc on noisy device
                        self.trained_network[str_NNs]['reward'] = reward

                    # HW1, RC = self.para2interface_HW(Para_HW1) # TODO: hardware
                    # HW_Eff = self.get_HW_efficiency(Network, HW1, RC) # TODO: hardware

                    # # HW Efficiency
                    # logger.info('------>Hardware Efficiency Exploreation {}<------'.format(HW_Eff)) # TODO: hardware
                    # Dec. 22: Second loop: search hardware
                    # HW_Eff == -1 indicates that violate the resource constriants
                    # # TODO: Hardware
                    # if HW_Eff == -1 or HW_Eff > self.target_HW_Eff:  # generate a network again
                    #     for i in range(controller_params["num_hw_per_child"]):
                    #         child_network, hyperparameters = self.generate_child_network(child_network)
                    #         l_Para_HW1 = hyperparameters[0][self.hw1_beg:self.hw1_end]
                    #
                    #         str_HW1 = " ".join(str(x) for x in l_Para_HW1)
                    #         str_HWs = str_HW1
                    #         DNA_HW1 = child_network[0][self.hw1_beg:self.hw1_end]
                    #
                    #         HW1, RC = self.para2interface_HW(Para_HW1)
                    #         HW_Eff = self.get_HW_efficiency(Network, HW1, RC)
                    #
                    #         if HW_Eff != -1:
                    #             logger.info('------>Hardware Exploreation  Success<------')
                    #             break
                    # else:
                    #     logger.info('------>No Need for Hardware Exploreation<------')

                    # if HW_Eff != -1 and HW_Eff <= self.target_HW_Eff: # TODO: Hardware
                    #     if str_NNs in self.trained_network.keys():  # used before
                    #             accuracy = self.trained_network[str_NNs]
                    #     else:
                    #         accuracy = random.uniform(0, 1)
                    #         # # Keep history trained data
                    #         # self.trained_network[str_NNs] = accuracy
                    #
                    #     # # norm_HW_Eff = (self.target_HW_Eff - HW_Eff) / self.target_HW_Eff    # TODO: Hardware
                    #     #
                    #     # # Weiwen 01-24: Set weight of HW Eff to 1 for hardware exploration only
                    #     # reward = max(accuracy * 0 + norm_HW_Eff * 1, -1)    # TODO: Hardware
                    #
                    #     # # TODO: Hardware
                    #     # # Help us to build the history table to avoid optimization for the same network
                    #     # # Weiwen 01-24: We comment this for exploration of hardware
                    #     # self.explored_info[str_NNs] = {}
                    #     # self.explored_info[str_NNs][0] = accuracy
                    #     # self.explored_info[str_NNs][1] = reward
                    #     # self.explored_info[str_NNs][2] = HW_Eff

                    # else:
                    #     accuracy = 0
                    #     reward = -1

                # still in the loop of a single sampled child network
                logger.info("====================Results=======================")
                logger.info("--------->NN: {}, Accuracy: {}".format(str_NNs, valid_acc))
                # logger.info("--------->HW: {}, Specs.: {}".format(str_HWs, HW_Eff))     # TODO: Hardware
                logger.info("--------->Reward: {}".format(reward))  # TODO: Do we have reward before?
                logger.info("=" * 50)

                # TODO(NOTE): record information, TODO: BS=1
                self.QNN_history.append(str_NNs)
                self.valid_acc_history.append(valid_acc)
                if args.Is_ideal_acc:
                    self.circuit_length_history.append(cir_len)

                episode_reward_buffer.append(reward)
                # identified_arch = np.array(list(DNA_NN1) + list(DNA_HW1))  # TODO: Hardware
                identified_arch = np.array(list(DNA_NN1))
                architecture_batch.append(identified_arch)

            # assemble all the sampled child networks in a specific episode
            current_reward = np.array(episode_reward_buffer)    # (BS, )

            # The reward_history is (n_current_episode), used for calculate the baseline reward
            mean_reward = np.mean(current_reward)   # average reward on batches
            self.reward_history.append(mean_reward)  # used for record, not for training
            self.architecture_history.append(child_network)  # used for record, not for training
            # total_rewards += mean_reward  # used for the reward in the whole episodes TODO: useless

            # TODO(NOTE): Reward is averaged by the baseline
            baseline = ema(self.reward_history)   # moving average across the episode
            last_reward = self.reward_history[-1]   # avg reward for current episode
            # rewards = current_reward - baseline
            rewards = [last_reward - baseline]  # TODO: BS=1

            # TODO(Note): update controller in each episode
            feed_dict = {
                self.child_network_paras: architecture_batch,
                self.batch_size: len(architecture_batch),
                self.discounted_rewards: rewards    # TODO: BS=1
            }

            with self.graph.as_default():
                _, _, loss, lr, gs = self.sess.run(
                    [self.train_operation, self.update_global_step, self.total_loss, self.learning_rate,
                     self.global_step], feed_dict=feed_dict)

            logger.info('=-=-=-=-=-=>Episode: {} | Loss: {} | LR: {} | Mean R: {} | Reward: {}<=-=-=-=-='.format(
                episode, loss, (lr, gs), mean_reward, rewards))

        print(self.reward_history)
        # self.plot_history(self.reward_history, ylim=(min(self.reward_history)-0.01, max(self.reward_history)-0.01))
        self.plot_history(self.reward_history, plt_filename,
                          ylim=(min(self.reward_history)-0.01, max(self.reward_history)+0.01))

        self.pareto_input = child_network  # TODO: BS=1
        # Another choice
        # self.pareto_input = np.array([[0] * self.num_para], dtype=np.int64)  # TODO: BS=1

    # TODO: The interface might be changed for the different training settings
    # TODO(Note): We only sample one child in the experiments
    def get_pareto_front(self, train_loader, test_loader, enc_circuit, qiskit_enc_circuit, pareto_args, args):
        # TODO(NOTE): sample QNN from the trained RNN controller
        # TODO(NOTE): Here we sample multiple child networks from the pdf of a singled run with a single input
        logger = self.logger
        num_child_QNN = args.num_child_QNN
        with self.graph.as_default():
            feed_dict = {
                self.child_network_paras: self.pareto_input,
                self.batch_size: 1  # TODO: BS=1
            }

        # [(bs, n_options_i), ..., ], length is n_time_step
        rnn_out = self.sess.run(self.RNN_pred_prob, feed_dict=feed_dict)    # TODO(NOTE): Run the RNN, BS=1
        DNA_sampled_child_QNN_list = []  # 2-d list, [[], [], ...], (num_child_QNN, n_time_step)
        for child_idx in range(num_child_QNN):
            DNA_sampled_child_QNN = []
            for layer_idx, prob in rnn_out.items():  # for each time step, prob with shape (bs=1, n_options_i)
                QNN_layer_idx = np.random.choice(range(len(self.para_2_val[layer_idx])), p=prob[0])  # TODO: BS=1
                DNA_sampled_child_QNN.append(QNN_layer_idx)
            DNA_sampled_child_QNN_list.append(DNA_sampled_child_QNN)

        # translate the index to the name of layer
        Para_sampled_child_QNN_list = []  # 2-d list, [[], [], ], (num_child_QNN, n_time_step)
        for child_idx in range(num_child_QNN):
            Para_sampled_child_QNN = []
            for layer_idx in range(self.num_para):  # for each time step, prob with shape (bs=1, n_options_i)
                sampled_layer_idx = DNA_sampled_child_QNN_list[child_idx][layer_idx]
                sampled_layer_name = self.para_2_val[layer_idx][sampled_layer_idx]
                Para_sampled_child_QNN.append(sampled_layer_name)

            self.QNN_history.append(Para_sampled_child_QNN)
            Para_sampled_child_QNN_list.append(Para_sampled_child_QNN)


        # TODO(Note): training each sampled QNN on complete training set and evaluate them on test set
        # acc_sampled_child_QNN_list = []
        # cir_len_sampled_child_QNN_list = []
        for child_idx in range(num_child_QNN):
            Para_child_QNN = Para_sampled_child_QNN_list[child_idx]
            Network = self.para2interface_NN(Para_child_QNN, args)
            if args.Is_ideal_acc:  # evaluate on ideal simulator
                # we need both the accuracy and the circuit length in this case
                # evaluate on test set
                # TODO: The interface might be changed for the different training settings
                test_acc, cir_len = QNN_Evaluation_Ideal(train_loader, test_loader, Network, pareto_args, args, logger,
                                                         enc_circuit=enc_circuit)
                self.valid_acc_history.append(test_acc)
                self.circuit_length_history.append(cir_len)
                # acc_sampled_child_QNN_list.append(test_acc)
                # cir_len_sampled_child_QNN_list.append(cir_len)

            else:
                # TODO(Note): validate the ideally trained model obtained with best epoch on the noisy device/simulator
                # TODO: The interface might be changed for the different training settings
                test_acc = QNN_Evaluation_Noise_QC(train_loader, test_loader, Network, pareto_args, args, logger,
                                                   enc_circuit=enc_circuit, qiskit_enc_circuit=qiskit_enc_circuit)
                # acc_sampled_child_QNN_list.append(test_acc)
                self.valid_acc_history.append(test_acc)

    def write_results_to_csv(self, args):
        if args.Is_ideal_acc:  # TODO: else for noisy
            # TODO(Note): Write the episode information to the log file
            # log_filename = "Experimental_Result/" + str(args.dataset) + "_" + str(args.interest_class) + "_" + str(
            #     args.file_flag) + ".csv"

            log_filename = "Experimental_Result/" + str(args.dataset) + "_" + str(args.interest_class) + "_beta_" + str(
                args.reward_beta) + "_encq_" + str(args.num_enc_qubits) + "_" + str(args.file_flag) + ".csv"

            search_dict = {}
            self.reward_history.append("None")  # TODO: BS=1,
            search_dict['episode_best_valid_acc'] = self.valid_acc_history
            search_dict['episode_circuit_length'] = self.circuit_length_history
            search_dict['episode_QNN_architecture'] = self.QNN_history
            search_dict['episode_reward'] = self.reward_history
            # print(len(self.valid_acc_history))
            # print(len(self.circuit_length_history))
            # print(len(self.QNN_history))
            # print(len(self.reward_history))

            search_dict_df = DataFrame(search_dict)
            search_dict_df.to_csv(log_filename, mode='a+', index=False)

            best_dict = {}
            best_dict['NAS_best_test_acc'] = [max(self.valid_acc_history)]
            best_dict['NAS_best_acc_episode'] = [self.valid_acc_history.index(max(self.valid_acc_history))]
            index = self.valid_acc_history.index(max(self.valid_acc_history))
            best_dict['NAS_best_acc_arch'] = [self.QNN_history[index]]
            best_dict['NAS_best_acc_cir_length'] = [self.circuit_length_history[index]]
            best_dict_df = DataFrame(best_dict)
            best_dict_df.to_csv(log_filename, mode='a+', index=False)

            # Print the dataframe
            print("=" * 20 + "The dataframe for NAS process is as follows" + "=" * 20)
            print(search_dict)
            print("=" * 20 + "The dataframe for NAS process is as follows" + "=" * 20)
            print(best_dict)


# %%
if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)     # TODO: Consider whether we need to set the random seed
    random.seed(seed)   # TODO: Consider whether we need to set the random seed
    logging.basicConfig(stream=sys.stdout,
                        level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')   # necessary?

    print("Begin")
    controller = Controller()   # Initialize
    controller.global_train()

    # TODO: Where to get the sampled NN? from the last episode?

    # TODO: Draw the figure using the final RNN controller?

# %%


