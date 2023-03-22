import torch
import math
import numpy as np
import random
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit import transpile
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer import AerSimulator


def num2bin(number, target_len):  # target_len is the predefined number of qubit (starting from 1)
    bin_str = bin(number).replace('0b', '')
    ori_len = len(bin_str)
    ext_len = target_len - ori_len
    bin_str = '0' * ext_len + bin_str  # zero extension

    # print(bin_str)  # It is a str!
    return bin_str


def get_prob_from_counts(counts):  # counts is a list of dict
    probs = []
    if isinstance(counts, dict):
        counts = [counts]   # counts is a list of dict

    for count in counts:
        total_shots = 0
        prob_dict = {}
        for key, value in count.items():
            total_shots += value
        for key, value in count.items():
            prob_dict[key] = value/total_shots
        probs.append(prob_dict)

    return probs  # still a list of dict


def get_prob_one_from_counts(counts, n_qubits):
    prob_ones = []
    if isinstance(counts, dict):
        counts = [counts]
    for count in counts:    # only one
        ctr_one = [0] * n_qubits
        total_shots = 0
        for k, v in count.items():
            # k = "{0:04b}".format(int(k, 16)) # 0x number based
            for qubit_idx in range(n_qubits):
                if k[qubit_idx] == '1':  # str[idx] -> char
                    ctr_one[qubit_idx] += v
            total_shots += v
        prob_one = np.array(ctr_one) / total_shots
        prob_ones.append(prob_one)
    res = np.stack(prob_ones)   # (1, n_qubits)
    return res


def IBM_Q_exe(qc, args):
    # TODO: Make it to run multiple jobs/times?

    """
    Running the quantum circuits on the real IBM quantum devices.
    :param qc: the quantum circuit to run before transpile
           backend_name: specify the name of backend to run
           num_shots: the desired number of shots to run
           optim_level: the optimization level for transpiling
    :return: output, (num_output), tensor
    """
    backend_name = args.backend_name
    num_shots = args.num_shots
    optim_level = args.optim_level

    n_output = args.output_num  # number of classes
    n_qubits = args.num_ori_qubits + args.num_enc_qubits  # total qubits

    # # TODO(NOTE): Load the provider (account)
    # print("*" * 30 + "IBM Quantum Processor handling start" + "*" * 30)
    # API_token = "3eec804fe4d033aaf06c2566f8b4711c16fd802f10dde0075d1bf307340489ca5ea6045075c7c185272f2a9566a7a1630a" \
    #             "9cd2c60d52c6e6dbd901438ee2e4c5"
    # hub = 'ibm-q-education'  # 'ibm-q'
    # group = 'george-mason-uni-1'  # 'open'
    # project = 'hardware-acceler'  # 'main'
    # IBMQ.save_account(token=API_token, hub=hub, group=group, project=project, overwrite=True)
    # provider = IBMQ.load_account()  # Load account from disk

    # TODO(NOTE): Get the appropriate backend
    provider = args.provider
    backend = provider.get_backend(backend_name)
    # selected_backend_list = provider.backends(filters=lambda x: x.configuration().n_qubits >= num_qubits
    #                                           and not x.configuration().simulator and x.status().operational == True)
    # backend = least_busy(selected_backend_list)

    # print("The selected backend is ", backend)
    config = backend.configuration()
    # print("The basis gates are ", config.basis_gates)
    shot_limit = config.max_shots
    # print("The shot limitation of this back end is ", shot_limit)
    num_shots = shot_limit if num_shots > shot_limit else num_shots
    # print("The actual number of shots is ", num_shots)
    # print("The maximum number of experiments of this backend is ", config.max_experiments)

    # TODO(NOTE): Run circuit on the backend
    # TODO: Retrieve job? why we need this? for historical?
    qc_trans = transpile(qc, backend, optimization_level=optim_level)
    job = backend.run(qc_trans, shots=num_shots)
    print("The job id is", job.job_id())
    job_monitor(job)
    counts = job.result().get_counts()  # e.g., {'00': 244, '01': 299, '10': 218, '11': 263}

    # ran_job = None
    # for ran_job in backend.jobs(limit=5):
    #     print(str(ran_job.job_id()) + " " + str(ran_job.status()))
    #     if ran_job is not None:
    #         job = backend.retrieve_job(ran_job.job_id())

    # status = job.status()
    if n_output <= n_qubits:
        # use the probability of |1> for each single qubit!
        qiskit_output = get_prob_one_from_counts(counts, n_qubits)  # (1, n_qubits)
        qiskit_output = qiskit_output[0]  # narray (n_qubits)
        output = []
        # for idx in range(n_qubits-1, n_qubits-1-n_output, -1):
        for offset in range(n_qubits):
            idx = n_qubits - 1 - offset
            output.append(qiskit_output[idx])

    elif n_output <= 2 ** n_qubits:
        # TODO(Note): Get the probability
        probs = get_prob_from_counts(counts)  # a list of dict
        probs = probs[0]    # prob is a dict
        # print("The complete probability distribution is", probs)
        output = []
        # TODO(Note): the circuit is in qiskit order. so handle the output in wiki order for consistency
        for amp_idx in range(n_output):
            amp_idx_str = num2bin(amp_idx, n_qubits)
            amp_idx_str = amp_idx_str[::-1]  # reverse for wiki order
            amp_idx_prob = probs[amp_idx_str]
            output.append(amp_idx_prob)
    else:
        raise Exception("The number of classes is larger than the number of amplitudes")

    output = torch.Tensor(output)

    return output   # (num_output), tensor


def Noisy_Aer_exe(qc, args):
    # TODO: Make it to run multiple jobs/times?

    """
    Running the quantum circuits on the noisy Aer simulators.
    :param qc: the quantum circuit to run before transpile
           backend_name: specify the name of backend to run
           num_shots: the desired number of shots to run
           optim_level: the optimization level for transpiling
    :return: output, (num_output), tensor
    """

    backend_name = args.backend_name
    num_shots = args.num_shots
    optim_level = args.optim_level

    n_output = args.output_num  # number of classes
    n_qubits = args.num_ori_qubits + args.num_enc_qubits  # total qubits

    # # TODO(NOTE): Load the provider (account)
    # print("*" * 30 + "IBM Quantum Processor handling start" + "*" * 30)
    # API_token = "3eec804fe4d033aaf06c2566f8b4711c16fd802f10dde0075d1bf307340489ca5ea6045075c7c185272f2a9566a7a1630a" \
    #             "9cd2c60d52c6e6dbd901438ee2e4c5"
    # hub = 'ibm-q-education'  # 'ibm-q'
    # group = 'george-mason-uni-1'  # 'open'
    # project = 'hardware-acceler'  # 'main'
    # IBMQ.save_account(token=API_token, hub=hub, group=group, project=project, overwrite=True)
    # provider = IBMQ.load_account()  # Load account from disk

    # TODO(NOTE): Get the appropriate backend
    provider = args.provider
    backend = provider.get_backend(backend_name)   # like 'ibmq_quito'
    # selected_backend_list = provider.backends(filters=lambda x: x.configuration().n_qubits >= num_qubits
    #                                           and not x.configuration().simulator and x.status().operational == True)
    # backend = least_busy(selected_backend_list)

    # print("The selected backend is ", backend)
    config = backend.configuration()
    # print("The basis gates are ", config.basis_gates)
    shot_limit = config.max_shots
    # print("The shot limitation of this back end is ", shot_limit)
    num_shots = shot_limit if num_shots > shot_limit else num_shots
    # print("The actual number of shots is ", num_shots)
    # print("The maximum number of experiments of this backend is ", config.max_experiments)

    # TODO(NOTE): Run circuit on the noisy simulator
    sim_backend = AerSimulator.from_backend(backend)

    qc_trans = transpile(qc, sim_backend, optimization_level=optim_level)
    job_sim = sim_backend.run(qc_trans, shots=num_shots)
    job_monitor(job_sim, interval=1)
    result_sim = job_sim.result()
    counts = result_sim.get_counts()  # e.g., {'00': 244, '01': 299, '10': 218, '11': 263}

    # ran_job = None
    # for ran_job in backend.jobs(limit=5):
    #     print(str(ran_job.job_id()) + " " + str(ran_job.status()))
    #     if ran_job is not None:
    #         job = backend.retrieve_job(ran_job.job_id())

    # status = job.status()
    if n_output <= n_qubits:
        # use the probability of |1> for each single qubit!
        qiskit_output = get_prob_one_from_counts(counts, n_qubits)  # (1, n_qubits)
        qiskit_output = qiskit_output[0]  # narray (n_qubits)
        output = []
        # for idx in range(n_qubits-1, n_qubits-1-n_output, -1):
        for offset in range(n_qubits):
            idx = n_qubits - 1 - offset
            output.append(qiskit_output[idx])

    elif n_output <= 2 ** n_qubits:
        # TODO(Note): Get the probability
        probs = get_prob_from_counts(counts)  # a list of dict
        probs = probs[0]    # prob is a dict
        # print("The complete probability distribution is", probs)
        output = []
        # TODO(Note): the circuit is in qiskit order. so handle the output in wiki order for consistency
        for amp_idx in range(n_output):
            amp_idx_str = num2bin(amp_idx, n_qubits)
            amp_idx_str = amp_idx_str[::-1]  # reverse for wiki order
            amp_idx_prob = probs[amp_idx_str]
            output.append(amp_idx_prob)
    else:
        raise Exception("The number of classes is larger than the number of amplitudes")

    output = torch.Tensor(output)

    return output   # (num_output), tensor