import torch
import utils
import argparse
import check_point
import model
import scipy.sparse as sp
import numpy as np
import data_utils

parser = argparse.ArgumentParser(description='Generate predict')
parser.add_argument('--ckpt_dir', type=str, help='folder contains check point', required=True)
parser.add_argument('--model_name', type=str, help='name of model', required=True)
parser.add_argument('--epoch', type=int, help='last epoch before interrupt', required=True)
parser.add_argument('--data_dir', type=str, help='folder contains data', required=True)
parser.add_argument('--nb_hop', type=int, help='level of correlation matrix', default=1)
parser.add_argument('--batch_size', type=int, help='batch size predict', default=8)

args = parser.parse_args()

prefix_model_ckpt = args.model_name
ckpt_dir = args.ckpt_dir
data_dir = args.data_dir
real_adj_matrix = sp.load_npz(data_dir + 'adj_matrix/r_matrix_'+ str(args.nb_hop) + 'w.npz')

ckpt_path = ckpt_dir + '/' + prefix_model_ckpt + '/' + 'epoch_' + str(args.epoch) + '/' + prefix_model_ckpt + '_checkpoint.pt'
config_param_file = ckpt_dir + '/' + prefix_model_ckpt + '/' + prefix_model_ckpt + '_config.json'
load_param = check_point.load_config_param(config_param_file)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_data_type = torch.float32

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
print('density of C matrix: %.6f' % (real_adj_matrix.nnz * 1.0 / NB_ITEMS / NB_ITEMS))

batch_size = args.batch_size
# train_loader = data_utils.generate_data_loader(train_instances, load_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=True)
# valid_loader = data_utils.generate_data_loader(validate_instances, load_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)
test_loader = data_utils.generate_data_loader(test_instances, batch_size, item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)

pre_trained_model = model.RecSysModel(load_param, MAX_SEQ_LENGTH, item_probs, real_adj_matrix.todense(), device, model_data_type)
pre_trained_model.to(device, dtype= model_data_type)
optimizer = torch.optim.RMSprop(pre_trained_model.parameters(), lr= 0.001)

load_model, _, _, _, _, _, _, _, _ = check_point.load_ckpt(ckpt_path, pre_trained_model, optimizer)

def HLU_score_for_data(model, data_loader, batch_size):
    device = model.device
    nb_batch = len(data_loader.dataset) // batch_size
    if len(data_loader.dataset) % batch_size == 0:
        total_batch = nb_batch
    else :
        total_batch = nb_batch + 1
    print(total_batch)
    list_HLU_score = []
    C = 100
    beta = 5
    model.eval()
    for i, data_pack in enumerate(data_loader, 0):
        data_x, data_seq_len, data_y = data_pack
        x_ = data_x.to_dense().to(dtype=model.d_type, device=device)
        real_batch_size = x_.size()[0]
        hidden = model.init_hidden(real_batch_size)
        y_ = data_y.to(dtype=model.d_type, device=device)
        predict_ = model(x_, data_seq_len, hidden)
        sigmoid_pred = torch.sigmoid(predict_)
        sorted_rank, indices = torch.sort(sigmoid_pred, descending = True)
        for seq_idx, a_sorted_seq in enumerate(indices):
            sum_of_rank_score = 0

            #tinh tu so cua HLU
            for rank, idx in enumerate(a_sorted_seq):
                if(y_[seq_idx, idx] == 1.0):
                    rank_score = 2**((1-(rank+1))/(beta-1))
                    sum_of_rank_score += rank_score

            sum_of_rank_target_basket = 0
            # tinh mau so cua HLU
            target_basket_size = (y_[seq_idx] != 0.0).sum(dim=-1).item()
            for r in range(1, target_basket_size+1):
                target_rank_score = 2**((1-r)/(beta-1))
                sum_of_rank_target_basket += target_rank_score
            HLU_score = C * sum_of_rank_score / sum_of_rank_target_basket
            list_HLU_score.append(HLU_score)
    print("HLU list len: %d" % len(list_HLU_score))
    return np.array(list_HLU_score).mean()

avg_HLU_score = HLU_score_for_data(load_model, test_loader, batch_size)
print("HLU score: %.6f" % avg_HLU_score)