
import scipy.sparse as sp
import os, re

################## build knowledge about data ###################

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