import os
import pandas as pd
import time

# dataset related parameters
dataset = "mnist_general"  # "mnist"  # "fmnist"  #
interest_class = "3,6"  #   "0,3,6"  #   "0,3,6,9"  #
img_size = "4"  # "8"  #
img_size_col = "2"  # "4"  # "8"  #
num_ori_qubits = "3"  # "4"  # "6"  #
backend_name = "ibmq_manila"  # "ideal_sim"  # "ibmq_belem" #

seed = "2"
optim_level = "2"  # "3"
n_ts_per_class = "50"  # "0"  #
batch_size = '64'

# max_episodes = "100"

# TODO(NOTE): special care of mnist-4
class_list = [int(x.strip()) for x in interest_class.split(",")]
num_classes = len(class_list)

# prepare necessary input list, all the elements are str already!
# elements among lists are pair-wise

# list_NAS_best_acc_arch = []
list_reward_beta = []   # TODO(NOTE): Just used for counting
list_num_enc_qubits = []
list_ry_angle_factor_list = []
list_entag_pattern = []
list_permutation_list = []
list_basic_path_list = []

Is_ideal_acc = False  # True     #
Is_real_qc = True   #   False  #

# assert Is_ideal_acc + Is_real_qc == 1, "error in setting the backend!"
input_pre = "Added_Readin/"
output_pre = "Added_Output/"
if Is_ideal_acc:
    input_dir = input_pre + "Basic_model/Ideal/"
    output_dir = output_pre + "Basic_model/Ideal/"
elif Is_real_qc:
    input_dir = input_pre + "Basic_model/Real_qc/"
    output_dir = output_pre + "Basic_model/Real_qc/"
else:
    input_dir = input_pre + "Basic_model/Noisy_sim/"
    output_dir = output_pre + "Basic_model/Noisy_sim/"

Input_filename = input_dir + dataset + "_" + interest_class + "_" + num_ori_qubits + "_oriq_" + backend_name + \
                 "_pristiq_input.csv"

df = pd.read_csv(Input_filename)

# for row in df["NAS_best_acc_arch"]:
#     row = ",".join(row.split())
#     list_NAS_best_acc_arch.append(row)  # v2,v0,v2,v5,v5

for row in df["reward_beta"]:  # int
    if not isinstance(row, str):
        row = str(row)
    list_reward_beta.append(row)

for row in df["num_enc_qubits"]:  # int
    if not isinstance(row, str):
        row = str(row)
    list_num_enc_qubits.append(row)

for row in df["ry_angle_factor_list"]:
    row = row.strip("[").strip("]").replace(" ", "")
    list_ry_angle_factor_list.append(row)

for row in df["entag_pattern"]:
    list_entag_pattern.append(row)

for row in df["permutation_list"]:
    if row != "None":
        row = [x.strip(' [').strip('[').strip(']').replace(" ", "") for x in row.split("],")]
        row = str(row)
        row = row.replace(" ", "")
    list_permutation_list.append(row)

for row in df['basic_model_path']:
    if not isinstance(row, str):
        row = str(row)
    list_basic_path_list.append(row)

# whether w/ expanded is not related to this func
if Is_ideal_acc:
    for index in range(len(list_reward_beta)):
        if num_classes == 4:
            pass
        else:
            pass
        # print("The command is, ", cmd)
        # os.system(cmd)

elif Is_real_qc:    # real qc
    for index in range(len(list_reward_beta)):
        if num_classes == 4:
            pass
        else:
            cmd = 'sbatch ADD_secure_template_real_qc.sh ' + dataset + ' ' + interest_class + ' ' + img_size + ' ' \
                  + img_size_col + ' ' + batch_size + ' ' + batch_size + ' ' + n_ts_per_class + ' ' + num_ori_qubits \
                  + ' ' + list_num_enc_qubits[index] + ' ' + list_ry_angle_factor_list[index] + ' ' \
                  + list_entag_pattern[index] + ' ' + list_permutation_list[index] + ' ' + list_basic_path_list[index] \
                  + ' ' + backend_name + ' ' + seed + ' ' + optim_level + ' ' + output_dir

        print("In the original python file, ", cmd)
        os.system(cmd)
        time.sleep(60)

else:   # noisy simulator
    for index in range(len(list_reward_beta)):
        if num_classes == 4:
            pass
        else:
            pass
        # print("The command is, ", cmd)
        # os.system(cmd)

print("finished!")

