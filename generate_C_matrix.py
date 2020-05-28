import argparse
import utils
import scipy.sparse as sp

parser = argparse.ArgumentParser(description='Generate C matrix.')

parser.add_argument('--nb_hop', help='The order of the real adjacency matrix (default:1)', default=1)
parser.add_argument('--data_dir', help='Data folder', required=True)
args = parser.parse_args()

data_dir = args.data_dir
output_dir = data_dir + '/adj_matrix'
nb_hop = args.nb_hop

train_data_path = data_dir + 'train.txt'
train_instances = utils.read_instances_lines_from_file(train_data_path)
nb_train = len(train_instances)
print(nb_train)

validate_data_path = data_dir + 'validate.txt'
validate_instances = utils.read_instances_lines_from_file(validate_data_path)
nb_validate = len(validate_instances)
print(nb_validate)

test_data_path = data_dir + 'test.txt'
test_instances = utils.read_instances_lines_from_file(test_data_path)
nb_test = len(test_instances)
print(nb_test)

### build knowledge ###

print("@Build knowledge")
MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs = utils.build_knowledge(train_instances, validate_instances)

print("#Statistic")
NB_ITEMS = len(item_dict)
print(" + Maximum sequence length: ", MAX_SEQ_LENGTH)
print(" + Total items: ", NB_ITEMS)

print("@Build the real adjacency matrix")
real_adj_matrix = utils.build_sparse_adjacency_matrix_v2(train_instances, validate_instances, item_dict)
real_adj_matrix = utils.normalize_adj(real_adj_matrix)


##### calculate correlatoin matrix ######
rmatrix_fpath = output_dir + "/r_matrix_" + str(nb_hop) + "w.npz"
mul = real_adj_matrix
w_mul = real_adj_matrix
coeff = 1.0
for w in range(1, 1):
    coeff *= 0.85
    w_mul *= real_adj_matrix
    w_mul = utils.remove_diag(w_mul)

    w_adj_matrix = utils.normalize_adj(w_mul)
    mul += coeff * w_adj_matrix

real_adj_matrix = mul
print('density : %.6f' % (real_adj_matrix.nnz * 1.0 / NB_ITEMS / NB_ITEMS))
sp.save_npz(rmatrix_fpath, real_adj_matrix)
print(" + Save adj_matrix to" + rmatrix_fpath)