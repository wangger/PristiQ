import torch
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from c_mnist import *
from SelfMNIST import *


def modify_target_ori(target, interest_num):
    # TODO(NOTE): Map the label, e,g [3, 6, 7] -> [0, 1 ,2]
    # TODO: Does it matter? If not build one-hot, seems does not matter? It might be related to the loss calculation
    # The implementation is not very efficient

    for j in range(len(target)):
        for idx in range(len(interest_num)):
            if target[j] == interest_num[idx]:
                target[j] = idx
                break

    # TODO(NOTE): Build one-hot label
    new_target = torch.zeros(target.shape[0], len(interest_num))

    for i in range(target.shape[0]):
        one_shot = torch.zeros(len(interest_num))
        one_shot[target[i].item()] = 1
        new_target[i] = one_shot.clone()

    # targets is narray, new_target is tensor
    return target, new_target


def modify_target(target, interest_num):
    new_target = torch.zeros(target.shape[0], len(interest_num))

    for i in range(target.shape[0]):
        one_shot = torch.zeros(len(interest_num))
        one_shot[target[i].item()] = 1
        new_target[i] = one_shot.clone()
    return target, new_target


def select_num(dataset, interest_num):
    """
    Function select_num is to select the specfic label of a dataset to generate a sub-dataset.
    Args:
         dataset: a pytorch-datasets
         interest_num: the specific label list,such as [3,6]
    Returns:
        dataset: a sub-dataset, The output target will be re-organized to [0, 1] (for [3,6])
    """
    labels = dataset.targets  # get labels
    labels = labels.numpy()
    idx = {}
    for num in interest_num:
        idx[num] = np.where(labels == num)

    fin_idx = idx[interest_num[0]]
    for i in range(1, len(interest_num)):
        fin_idx = (np.concatenate((fin_idx[0], idx[interest_num[i]][0])),)

    fin_idx = fin_idx[0]

    dataset.targets = labels[fin_idx]
    dataset.data = dataset.data[fin_idx]

    # print(dataset.targets.shape)

    dataset.targets, _ = modify_target_ori(dataset.targets, interest_num)
    # TODO(NOTE): Convert numpy to tensor!!!
    dataset.targets = torch.from_numpy(dataset.targets)
    # print(dataset.targets.shape)

    return dataset


class ToQuantumData_Batch(object):

    def __call__(self, tensor):
        data = tensor
        input_vec = data.view(-1)
        vec_len = input_vec.size()[0]
        input_matrix = torch.zeros(vec_len, vec_len)
        input_matrix[0] = input_vec
        input_matrix = input_matrix.transpose(0, 1)
        u, s, v = np.linalg.svd(input_matrix)
        output_matrix = torch.tensor(np.dot(u, v))
        output_data = output_matrix[:, 0].view(data.shape)
        return output_data


class ToQuantumData(object):
    """
    class ToQuantumData is to transform the input image into a unitary matrix to be encoded in quantum unitary matrix.
    """
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size

    def __call__(self, tensor):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = tensor.to(device)
        input_vec = data.view(-1)
        vec_len = input_vec.size()[0]
        input_matrix = torch.zeros(vec_len, vec_len)
        input_matrix[0] = input_vec
        input_matrix = input_matrix.transpose(0, 1)
        u, s, v = np.linalg.svd(input_matrix)
        output_matrix = torch.tensor(np.dot(u, v))
        # print(output_matrix)
        output_data = output_matrix[:, 0].view(1, self.img_size, self.img_size)
        return output_data


def load_data_mnist(interest_num, args, is_to_q=True, is_shuffle=True, disable_visualize=True):
    # Get the necessary input parameters
    img_size = int(args.img_size)
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    inference_batch_size = int(args.inference_batch_size)
    isppd = args.preprocessdata
    datapath = args.datapath

    if isppd:
        train_data = SelfMNIST(root=datapath, img_size=img_size, train=True)
        test_data = SelfMNIST(root=datapath, img_size=img_size, train=False)

    else:
        # Convert data to torch.FloatTensor
        if is_to_q:
            # TODO (NOTE): encode the data by amplitude encoding
            # TODO: Why does the input data has no normalization?? since the resize?
            transform = transforms.Compose(
                [transforms.Resize((img_size, img_size)), transforms.ToTensor(), ToQuantumData(img_size)])
            transform_inference = transforms.Compose(
                [transforms.Resize((img_size, img_size)), transforms.ToTensor(), ToQuantumData(img_size)])

        else:
            transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
            transform_inference = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

        # choose the training and test datasets
        train_data = datasets.MNIST(root=datapath, train=True,
                                    download=True, transform=transform)
        test_data = datasets.MNIST(root=datapath, train=False,
                                   download=True, transform=transform_inference)
    # print(len(train_data))
    # print(len(test_data))

    # the label of train_data and test_data is consistent
    # print("The select number is {}".format(interest_num))
    if disable_visualize:
        train_data = select_num(train_data, interest_num)  # TODO: Highlight
        test_data = select_num(test_data, interest_num)
    else:
        pass  # TODO(Note): to use MNIST for visualizing the images, do not group the label of the same class

    # print(len(train_data))
    # print(len(test_data))

    # prepare data loaders
    # TODO: Why drop last?
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=is_shuffle, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=inference_batch_size,
                                              num_workers=num_workers, shuffle=False, drop_last=True)

    return train_loader, test_loader


def load_data_fmnist(interest_num, args, is_to_q=True, is_shuffle=True, disable_visualize=True):
    # Get the necessary input parameters
    img_size = int(args.img_size)
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    inference_batch_size = int(args.inference_batch_size)
    datapath = args.datapath    # same as mnist

    # Convert data to torch.FloatTensor
    if is_to_q:
        # TODO (NOTE): encode the data by amplitude encoding
        # TODO: Why does the input data has no normalization?? since the resize?
        transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor(), ToQuantumData(img_size)])
        transform_inference = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor(), ToQuantumData(img_size)])

    else:
        transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        transform_inference = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

    # choose the training and test datasets
    train_data = datasets.FashionMNIST(root=datapath, train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root=datapath, train=False, download=True, transform=transform_inference)
    # print(len(train_data))
    # print(len(test_data))

    # the label of train_data and test_data is consistent
    # print("The select number is {}".format(interest_num))
    if disable_visualize:
        train_data = select_num(train_data, interest_num)  # TODO: Highlight
        test_data = select_num(test_data, interest_num)
    else:
        pass  # TODO(Note): to use MNIST for visualizing the images, do not group the label of the same class

    # print(len(train_data))
    # print(len(test_data))

    # prepare data loaders
    # TODO: Why drop last?
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=is_shuffle, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=inference_batch_size,
                                              num_workers=num_workers, shuffle=False, drop_last=True)

    return train_loader, test_loader




# def load_PSNR_data(interest_num, args):  # TODO(NOTE): To ensure the correct order, do not support pdd dataset here
#     # Get the necessary input parameters
#     img_size = int(args.img_size)
#     batch_size = int(args.batch_size)
#     num_workers = int(args.num_workers)
#     inference_batch_size = int(args.inference_batch_size)
#     datapath = args.datapath
#
#     # Convert data to torch.FloatTensor
#     # TODO (NOTE): encode the data by amplitude encoding
#     # TODO: Why does the input data have no normalization?? since the resize?
#     transform_amp = transforms.Compose(
#         [transforms.Resize((img_size, img_size)), transforms.ToTensor(), ToQuantumData(img_size)])
#     transform_amp_inference = transforms.Compose(
#         [transforms.Resize((img_size, img_size)), transforms.ToTensor(), ToQuantumData(img_size)])
#
#     transform_clean = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
#     transform_clean_inference = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
#
#     # choose the training and test datasets
#     train_data_amp = datasets.MNIST(root=datapath, train=True,
#                                     download=True, transform=transform_amp)
#     test_data_amp = datasets.MNIST(root=datapath, train=False,
#                                    download=True, transform=transform_amp_inference)
#
#     train_data_clean = datasets.MNIST(root=datapath, train=True,
#                                       download=True, transform=transform_clean)
#     test_data_clean = datasets.MNIST(root=datapath, train=False,
#                                      download=True, transform=transform_clean_inference)
#
#     # the label of train_data and test_data is consistent
#     train_data_amp = select_num(train_data_amp, interest_num)
#     test_data_amp = select_num(test_data_amp, interest_num)
#
#     train_data_clean = select_num(train_data_clean, interest_num)
#     test_data_clean = select_num(test_data_clean, interest_num)
#
#     # Merge the two dataset
#     train_data = torch.utils.data.TensorDataset(train_data_clean, train_data_amp)
#     test_data = torch.utils.data.TensorDataset(test_data_clean, test_data_amp)
#
#     # prepare the dataloader
#     # TODO: Why drop last?
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
#                                                num_workers=num_workers, shuffle=True, drop_last=True)
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size=inference_batch_size,
#                                               num_workers=num_workers, shuffle=False, drop_last=True)
#
#     return train_loader, test_loader


def to_quantum_matrix(tensor):
    """
    Function to_quantum_matrix is to transform the input image into a unitary matrix.
    Args:
         tensor: input image
    Returns:
        output_matrix: a unitary matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = tensor.to(device)
    input_vec = data.view(-1)
    vec_len = input_vec.size()[0]
    input_matrix = torch.zeros(vec_len, vec_len)
    input_matrix[0] = input_vec
    input_matrix = np.float64(input_matrix.transpose(0, 1))
    u, s, v = np.linalg.svd(input_matrix)
    output_matrix = torch.tensor(np.dot(u, v), dtype=torch.float64)
    return output_matrix