# %%
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import copy
import argparse
import time
import torch.nn.functional as F

class ToQuantumData(object):
    def __call__(self, tensor):
        # torch.set_printoptions(profile="full")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = tensor.to(device)
        input_vec = data.view(-1)

        input_vec = input_vec.to(device)
        vec_len = input_vec.size()[0]
        input_matrix = torch.zeros(vec_len, vec_len)
        input_matrix = input_matrix.to(device)
        input_matrix[0] = input_vec
        input_matrix = input_matrix.transpose(0, 1)

        u, s, v = torch.svd(input_matrix)
        u = u.to(device)
        v = v.to(device)
        output_matrix = u.matmul(v)
        # print(output_matrix.shape)
        # u, s, v = np.linalg.svd(input_matrix)
        # output_matrix = torch.tensor(np.dot(u, v))
        # print(output_matrix.shape)
        # sys.exit(0)
        output_data = output_matrix[:, 0].view(tensor.shape)
        return output_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QuantumFlow Qiskit Simulation')
    parser.add_argument('-s','--size',default="4",help="image resize",)
    parser.add_argument('-t', "--test", help="Only Test without Training", action="store_true", )
    parser.add_argument('-b', "--batch", default="100", help="batchsize", )
    args = parser.parse_args()

    print("=" * 100)
    print("Data Convert for MNIST. This script is for converting data to unitray matrix.")
    print("\tStart at:", time.strftime("%m/%d/%Y %H:%M:%S"))
    print("\tProblems and issues, please contact Dr. Weiwen Jiang (wjiang2@nd.edu)")
    print("\tEnjoy and Good Luck!")
    print("=" * 100)
    print()

    size = int(args.size)
    istest = args.test

    data_str = "training"
    if istest:
        data_str = "test"

    res = torch.load("processed/"+data_str+".pt")
    print(res[0].shape, res[1].shape)

    trans_to_tensor = transforms.ToTensor()
    trans_to_qc_data = ToQuantumData()
    trans_resize = transforms.Resize((size, size))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(res[0].shape[0]):
        if i%500==0:
            print(i)
        npimg = res[0][i].numpy()
        im = Image.fromarray(npimg, mode="L")
        qc_data = trans_to_tensor(trans_resize(im))[0]
        qc_data = qc_data.unsqueeze(0)

        if i==0:
            resize_float_data = qc_data.detach().clone()
        else:
            resize_float_data = torch.cat((resize_float_data, qc_data.detach().clone()))
    resize_float_data = resize_float_data.to(device)

    print("Resize Done!")

    # print(qc_data)
    #
    #
    # npimg = float_data[0][0].numpy()
    # im = Image.fromarray(npimg,mode="F")
    # qc_data = trans_to_tensor(trans_resize(im))[0]
    # print(qc_data)
    #
    # resize_float_data = F.interpolate(float_data, size, mode="bilinear")
    # # resize_float_data = resize_float_data.squeeze(0)
    #
    # print(resize_float_data[0][0])
    # sys.exit(0)



    batch_size = int(args.batch)
    batch_num = int(resize_float_data.shape[0]/batch_size)

    # data = torch.zeros(res[0].shape[0], size, size, dtype=torch.float32)
    # data = data.to(device)


    for i in range(batch_num):
        print(i)
        batch_data = resize_float_data[batch_size*i:batch_size*(i+1),:,:]
        data = batch_data.to(device)
        batch_size = batch_data.shape[0]
        input_vec = data.view(batch_size,-1)
        input_vec = input_vec.to(device)
        vec_len = input_vec.size()[1]
        input_vec = input_vec.unsqueeze(2)
        col_zeros = torch.zeros(batch_size,vec_len,vec_len-1)
        col_zeros = col_zeros.to(device)
        input_vec = torch.cat([input_vec, col_zeros], dim=-1)

        u, s, v = torch.svd(input_vec)

        u = u.to(device)
        v = v.to(device)
        output_matrix = u.matmul(v)
        output_data = output_matrix[:, :, 0].view(batch_data.shape)

        if i==0:
            fin_data = output_data.detach().clone()
        else:
            fin_data = torch.cat((fin_data, output_data.detach().clone()))

    # %%
    x = (fin_data, res[1])
    torch.save(x, 'processed/qc_'+data_str+'_'+str(size)+'_'+str(size)+'.pt')
    print("Convert Done!!!!")
    print("stored at:",'processed/'+data_str+'_'+str(size)+'_'+str(size)+'.pt')


