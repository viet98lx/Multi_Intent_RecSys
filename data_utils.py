import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import re
####################  Pre-process data   #######################

def create_binary_vector(item_list, item_dict):
    v = np.zeros(len(item_dict), dtype='int32')
    for item in item_list:
        v[item_dict[item]] = 1
    return v


def get_epoch(x):
    idx = x.index('_') + 1
    return int(x[idx:])


def compute_total_batches(nb_intances, batch_size):
    total_batches = int(nb_intances / batch_size)
    if nb_intances % batch_size != 0:
        total_batches += 1
    return total_batches


def seq_generator(raw_lines, item_dict):
    O = []
    S = []
    L = []
    Y = []

    lines = raw_lines[:]

    for line in lines:
        elements = line.split("|")

        # label = float(elements[0])
        bseq = elements[1:-1]
        tbasket = elements[-1]

        # Keep the length for dynamic_rnn
        L.append(len(bseq))

        # Keep the original last basket
        O.append(tbasket)

        # Add the target basket
        target_item_list = re.split('[\\s]+', tbasket)
        Y.append(create_binary_vector(target_item_list, item_dict))

        s = []
        for basket in bseq:
            item_list = re.split('[\\s]+', basket)
            id_list = [item_dict[item] for item in item_list]
            s.append(id_list)
        S.append(s)

    return {'S': np.asarray(S), 'L': np.asarray(L), 'Y': np.asarray(Y), 'O': np.asarray(O)}


def get_sparse_tensor_info(x, is_bseq=False):
    indices = []
    if is_bseq:
        for sid, bseq in enumerate(x):
            for t, basket in enumerate(bseq):
                for item_id in basket:
                    indices.append([sid, t, item_id])
    else:
        for bid, basket in enumerate(x):
            for item_id in basket:
                indices.append([bid, item_id])

    values = torch.ones(len(indices), dtype=torch.float32)
    indices = torch.IntTensor(indices)

    return indices, values

def generate_data_loader(data_instances, b_size, item_dict, max_seq_len, is_bseq, is_shuffle):
    data_seq = seq_generator(data_instances, item_dict)
    sparse_seq_indices, sparse_seq_values = get_sparse_tensor_info(data_seq['S'], is_bseq)
    sparse_seq_indices = torch.transpose(sparse_seq_indices, 0, 1)

    sparse_X = torch.sparse_coo_tensor(indices = sparse_seq_indices, values = sparse_seq_values,
                                       size=[len(data_seq['S']), max_seq_len, len(item_dict)])
    print(sparse_X)

    Y = torch.FloatTensor(data_seq['Y'])
    print(Y.shape)

    seq_len = torch.IntTensor(data_seq['L'])

    dataset = TensorDataset(sparse_X, seq_len, Y)
    data_loader = DataLoader(dataset=dataset, batch_size= b_size, shuffle= is_shuffle, num_workers=0)
    return data_loader