import model
import model_utils
import utils
import data_utils
import loss
import check_point

import argparse
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import scipy.sparse as sp
import random
import os
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generate C matrix.')

parser.add_argument('--batch_size', type=int, help='batch size of data set (default:32)', default=32)
parser.add_argument('--rnn_units', type=int, help='number units of hidden size lstm', default=16)
parser.add_argument('--rnn_layers', type=int, help='number layers of RNN', default=1)
parser.add_argument('--alpha', type=float, help='coefficient of C matrix in predict item score', default=0.4)
parser.add_argument('--lr', type=float, help='learning rate of optimizer', default=0.001)
parser.add_argument('--dropout', type=float, help='drop out after linear model', default= 0.3)
parser.add_argument('--embed_dim', type=int, help='dimension of linear layers', default=8)
parser.add_argument('--embed_dim_transformer', type=int, help='dimension of transformer project layers', default=8)
parser.add_argument('--transformer_layers', type=int, help='number transformer layers', default=1)
parser.add_argument('--transformer_head', type=int, help='number heads of transformer layers', default=2)
parser.add_argument('--device', type=str, help='device for train and predict', default='cpu')
parser.add_argument('--top_k', type=int, help='top k predict', default=10)
parser.add_argument('--nb_hop', type=int, help='top k predict', default=1)
parser.add_argument('--epoch', type=int, help='epoch to train', default=8)
parser.add_argument('--epsilon', type=float, help='different between loss of two consecutive epoch ', default=0.000005)
parser.add_argument('--model_name', type=str, help='name of model', required=True)
parser.add_argument('--data_dir', type=str, help='folder contains data', required=True)
parser.add_argument('--out_put_dir', type=str, help='folder to save model', required=True)

args = parser.parse_args()

config_param={}
config_param['rnn_units'] = args.rnn_units
config_param['rnn_layers'] = args.rnn_layers
config_param['dropout'] = args.dropout
config_param['embedding_dim'] = args.embed_dim
config_param['batch_size'] = args.batch_size
config_param['embed_transformer'] = args.embed_dim_transformer
config_param['num_heads'] = args.transformer_head
config_param['n_transformer_layers'] = args.transformer_layers
config_param['n_transformer_layers'] = args.transformer_layers
config_param['top_k'] = args.top_k

data_dir = args.data_dir
output_dir = args.out_put_dir
nb_hop = args.nb_hop

torch.manual_seed(1)
np.random.seed(2)
random.seed(0)

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

print("---------------------@Build knowledge-------------------------------")
MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs = utils.build_knowledge(train_instances, validate_instances)

print("#Statistic")
NB_ITEMS = len(item_dict)
print(" + Maximum sequence length: ", MAX_SEQ_LENGTH)
print(" + Total items: ", NB_ITEMS)

print('---------------------Load correlation matrix-------------------')

if (os.path.isfile(data_dir + 'adj_matrix/r_matrix_' +str(nb_hop)+ 'w.npz')):
    real_adj_matrix = sp.load_npz(data_dir + 'adj_matrix/r_matrix_1w.npz')
else:
    real_adj_matrix = sp.csr_matrix((NB_ITEMS, NB_ITEMS), dtype="float32")
print('Density of correlation matrix: %.6f' % (real_adj_matrix.nnz * 1.0 / NB_ITEMS / NB_ITEMS))

print('---------------------Create data loader--------------------')
train_loader = data_utils.generate_data_loader(train_instances, config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=True)
valid_loader = data_utils.generate_data_loader(validate_instances, config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)
test_loader = data_utils.generate_data_loader(test_instances, config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)

print('---------------------Create model------------------------')
print(args.device)
exec_device = torch.device('cuda' if (torch.cuda.is_available() and args.device != 'cpu') else 'cpu')
data_type = torch.float32
rec_sys_model = model.RecSysModel(config_param, MAX_SEQ_LENGTH, item_probs, real_adj_matrix.todense(), exec_device, data_type)
rec_sys_model.to(exec_device, dtype= data_type)

print('-----------------SUMMARY MODEL----------------')
pytorch_total_params = sum(p.numel() for p in rec_sys_model.parameters() if p.requires_grad)
print('number params: %d' % pytorch_total_params)
print(rec_sys_model)
for param in rec_sys_model.parameters():
  print(param.shape)

loss_func = loss.Weighted_BCE_Loss()
optimizer = torch.optim.RMSprop(rec_sys_model.parameters(), lr= args.lr, weight_decay= 5e-6)

try:
    os.makedirs(output_dir, exist_ok = True)
    print("Directory '%s' created successfully" % output_dir)
except OSError as error:
    print("Directory '%s' can not be created" % output_dir)

checkpoint_dir = output_dir + '/check_point/'
best_model_dir = output_dir + '/best_model_checkpoint/'
model_name = args.model_name


top_k = config_param['top_k']
train_display_step = 100
val_display_step = 20
test_display_step = 4
epoch = args.epoch

test_loss_min = 1000
test_recall_max = 0
epsilon = args.epsilon

train_losses = []
val_losses = []
train_recalls = []
val_recalls = []
test_losses = []
test_recalls = []

print('-------------------Start Training Model---------------------')

############################ Train Model #############################

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.75)

for ep in range(epoch):

    rec_sys_model, optimizer, avg_train_loss, avg_train_recall = model_utils.train_model(rec_sys_model, loss_func, optimizer, train_loader,
                                                                                         ep + 1, top_k, train_display_step)
    train_losses.append(avg_train_loss)
    train_recalls.append(avg_train_recall)

    avg_val_loss, avg_val_recall = model_utils.validate_model(rec_sys_model, loss_func, valid_loader,
                                                              ep + 1, top_k, val_display_step)
    val_losses.append(avg_val_loss)
    val_recalls.append(avg_val_recall)

    avg_test_loss, avg_test_recall = model_utils.test_model(rec_sys_model, loss_func, test_loader,
                                                            ep + 1, top_k, test_display_step)
    test_losses.append(avg_test_loss)
    test_recalls.append(avg_test_recall)

    scheduler.step()

    checkpoint = {
        'epoch': ep + 1,
        'valid_loss_min': avg_val_loss,
        'best_recall': avg_val_recall,
        'state_dict': rec_sys_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    # save checkpoint
    check_point.save_ckpt(checkpoint, False, model_name, checkpoint_dir, best_model_dir, ep)
    check_point.save_config_param(checkpoint_dir, model_name, config_param)

    if ((test_loss_min - avg_test_loss) / test_loss_min > epsilon and avg_test_recall > test_recall_max):
        print('Test loss decrease from ({:.5f} --> {:.5f}) '.format(test_loss_min, avg_test_loss))
        print('Can save model')
        check_point.save_ckpt(checkpoint, True, model_name, checkpoint_dir, best_model_dir, ep)
        check_point.save_config_param(best_model_dir, model_name, config_param)
        test_loss_min = avg_test_loss
        test_recall_max = avg_test_recall

    print('-' * 100)

with plt.style.context('seaborn-dark'):
    # plt.figure(figsize=(4,3))
    plt.plot(train_losses, label='Training')
    plt.plot(val_losses, label='Validation')
    plt.plot(test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(frameon=False)
    plt.show()

with plt.style.context('seaborn-dark'):
    # plt.figure(figsize=(4,3))
    plt.plot(train_recalls, label='Training ')
    plt.plot(val_recalls, label='Validation ')
    plt.plot(test_recalls, label='Test ')
    plt.xlabel('Epoch')
    plt.ylabel('Recall@%d' % top_k)
    plt.legend(frameon=False)
    plt.show()