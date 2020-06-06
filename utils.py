import scipy.sparse as sp
import numpy as np
import torch
import os, re
import itertools
import matplotlib
import matplotlib.pyplot as plt

################## utils and build knowledge about data ###################

def build_knowledge(training_instances, validate_instances):
    MAX_SEQ_LENGTH = 0
    item_freq_dict = {}

    for line in training_instances:
        elements = line.split("|")

        if len(elements) - 1 > MAX_SEQ_LENGTH:
            MAX_SEQ_LENGTH = len(elements) - 1

        if len(elements) == 3:
            basket_seq = elements[1:]
        else:
            basket_seq = [elements[-1]]

        for basket in basket_seq:
            item_list = re.split('[\\s]+', basket)
            for item_obs in item_list:
                if item_obs not in item_freq_dict:
                    item_freq_dict[item_obs] = 1
                else:
                    item_freq_dict[item_obs] += 1

    for line in validate_instances:
        elements = line.split("|")

        if len(elements) - 1 > MAX_SEQ_LENGTH:
            MAX_SEQ_LENGTH = len(basket_seq) - 1

        label = int(elements[0])
        if label != 1 and len(elements) == 3:
            basket_seq = elements[1:]
        else:
            basket_seq = [elements[-1]]

        for basket in basket_seq:
            item_list = re.split('[\\s]+', basket)
            for item_obs in item_list:
                if item_obs not in item_freq_dict:
                    item_freq_dict[item_obs] = 1
                else:
                    item_freq_dict[item_obs] += 1

    items = sorted(list(item_freq_dict.keys()))
    item_dict = dict()
    item_probs = []
    for item in items:
        item_dict[item] = len(item_dict)
        item_probs.append(item_freq_dict[item])

    item_probs = np.asarray(item_probs, dtype=np.float32)
    item_probs /= np.sum(item_probs)

    reversed_item_dict = dict(zip(item_dict.values(), item_dict.keys()))
    return MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs

def build_sparse_adjacency_matrix_v2(training_instances, validate_instances, item_dict):
    NB_ITEMS = len(item_dict)

    pairs = {}
    for line in training_instances:
        elements = line.split("|")

        if len(elements) == 3:
            basket_seq = elements[1:]
        else:
            basket_seq = [elements[-1]]

        for basket in basket_seq:
            item_list = re.split('[\\s]+', basket)
            id_list = [item_dict[item] for item in item_list]

            for t in list(itertools.product(id_list, id_list)):
                add_tuple(t, pairs)

    for line in validate_instances:
        elements = line.split("|")

        label = int(elements[0])
        if label != 1 and len(elements) == 3:
            basket_seq = elements[1:]
        else:
            basket_seq = [elements[-1]]

        for basket in basket_seq:
            item_list = re.split('[\\s]+', basket)
            id_list = [item_dict[item] for item in item_list]

            for t in list(itertools.product(id_list, id_list)):
                add_tuple(t, pairs)

    return create_sparse_matrix(pairs, NB_ITEMS)

def add_tuple(t, pairs):
    assert len(t) == 2
    if t[0] != t[1]:
        if t not in pairs:
            pairs[t] = 1
        else:
            pairs[t] += 1

def create_sparse_matrix(pairs, NB_ITEMS):
    row = [p[0] for p in pairs]
    col = [p[1] for p in pairs]
    data = [pairs[p] for p in pairs]
    adj_matrix = sp.csc_matrix((data, (row, col)), shape=(NB_ITEMS, NB_ITEMS), dtype="float32")
    nb_nonzero = len(pairs)
    density = nb_nonzero * 1.0 / NB_ITEMS / NB_ITEMS
    print("Density: {:.6f}".format(density))

    return sp.csc_matrix(adj_matrix, dtype="float32")

def create_identity_matrix(nb_items):
    return sp.identity(nb_items, dtype="float32").tocsr()

def create_zero_matrix(nb_items):
    return sp.csr_matrix((nb_items, nb_items), dtype="float32")

def normalize_adj(adj_matrix):
    """Symmetrically normalize adjacency matrix."""
    row_sum = np.array(adj_matrix.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_matrix = adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return normalized_matrix.tocsr()

def remove_diag(adj_matrix):
    new_adj_matrix = sp.csr_matrix(adj_matrix)
    new_adj_matrix.setdiag(0.0)
    new_adj_matrix.eliminate_zeros()
    return new_adj_matrix

def read_instances_lines_from_file(file_path):
    with open(file_path, "r") as f:
        lines = [line.rstrip('\n') for line in f]
        return lines

def predict_top_k(logits, top_k, batch_size, device, nb_items):
    predict_prob = torch.sigmoid(logits)
    # predict_prob = logits
    row_index = [i for i in range(0, batch_size)]
    top_k_col_indices = predict_prob.topk(dim=-1, k=top_k, sorted=True).indices.reshape([-1]).to(device)
    # print('---------------col indices --------------')
    # print(top_k_col_indices)
    top_k_row_indices = torch.ones(batch_size, top_k, dtype=torch.long) * torch.Tensor(row_index).type(
        torch.long).unsqueeze(1)
    top_k_row_indices = top_k_row_indices.reshape([-1]).to(device)
    # print('---------------row indices --------------')
    # print(top_k_row_indices)
    top_k_values = torch.ones(batch_size * top_k).to(device, logits.dtype)
    top_k_indices = torch.stack([top_k_row_indices, top_k_col_indices], dim=0)
    # print('---------------top k indices --------------')
    # print(top_k_indices)
    predict_top_k = torch.sparse_coo_tensor(indices=top_k_indices, values=top_k_values, size=(batch_size, nb_items))
    return predict_top_k.to_dense().type(logits.dtype)


def compute_recall_at_top_k(model, logits, top_k, target_basket, batch_size, device):
    nb_items = model.nb_items
    predict_basket = predict_top_k(logits, top_k, batch_size, device, nb_items)
    correct_predict = predict_basket * target_basket
    nb_correct = (correct_predict == 1.0).sum(dim=-1)
    actual_basket_size = (target_basket == 1.0).sum(dim=-1)
    # print(nb_correct)
    # print(actual_basket_size)

    return torch.mean(nb_correct.type(logits.dtype) / actual_basket_size.type(logits.dtype)).item()

def plot_loss(train_losses, val_losses):
    with plt.style.context('seaborn-dark'):
        plt.figure(figsize=(4,3))
        plt.plot(train_losses, label='Training')
        plt.plot(val_losses, label='Validation')
        # plt.plot(test_losses, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(frameon=False)
        plt.show()

def plot_recall(train_recalls, val_recalls):
    with plt.style.context('seaborn-dark'):
        plt.figure(figsize=(4,3))
        plt.plot(train_recalls, label='Training ')
        plt.plot(val_recalls, label='Validation ')
        # plt.plot(test_recalls, label='Test ')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend(frameon=False)
        plt.show()