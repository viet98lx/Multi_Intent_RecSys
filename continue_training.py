import argparse
import random
import torch
import numpy as np
import scipy.sparse as sp
import os
import matplotlib.pyplot as plt


import utils
import data_utils
import check_point
import model
import model_utils
import loss

torch.set_printoptions(precision=8)
parser = argparse.ArgumentParser(description='Continue training model')

parser.add_argument('--ckpt_dir', type=str, help='folder contains check point', required=True)
parser.add_argument('--epoch', type=int, help='number epoch to train', required=True)
parser.add_argument('--model_name', type=str, help='name of model', required=True)
parser.add_argument('--data_dir', type=str, help='folder contains data', required=True)
parser.add_argument('--output_dir', type=str, help='folder to save model', required=True)
parser.add_argument('--config_param_path', type=str, help='folder to save config param', required=True)
parser.add_argument('--lr', type=float, help='learning rate of optimizer', default=0.01)
# parser.add_argument('--top_k', type=int, help='top k predict', default=10)
parser.add_argument('--epsilon', type=float, help='different between loss of two consecutive epoch ', default=0.00000001)
parser.add_argument('--nb_hop', type=int, help='nb-hop', default=1)
parser.add_argument('--device', type=str, help='device for train and predict', default='cpu')

args = parser.parse_args()

epoch = args.epoch
model_name = args.model_name
data_dir = args.data_dir
output_dir = args.output_dir
ckpt_dir = args.ckpt_dir
best_ckpt_dir = output_dir + '/best_model_checkpoint'
nb_hop = args.nb_hop
config_param_file = args.config_param_path

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
    real_adj_matrix = sp.load_npz(data_dir + 'adj_matrix/r_matrix_' +str(nb_hop)+ 'w.npz')
else:
    real_adj_matrix = sp.csr_matrix((NB_ITEMS, NB_ITEMS), dtype="float32")
print('Density of correlation matrix: %.6f' % (real_adj_matrix.nnz * 1.0 / NB_ITEMS / NB_ITEMS))

print('---------------------Load check point-------------------')
ckpt_path = ckpt_dir + '/' + model_name + '/' + 'epoch_' + str(cur_epoch) + '/' + model_name + '_checkpoint.pt'

config_param = check_point.load_config_param(config_param_file)
exec_device = torch.device('cuda' if (torch.cuda.is_available() and args.device != 'cpu') else 'cpu')
data_type = torch.float32

pre_trained_model = model.RecSysModel(config_param, MAX_SEQ_LENGTH, item_probs, real_adj_matrix.todense(), exec_device, data_type)
pre_trained_model.to(exec_device, dtype= data_type)
optimizer = torch.optim.RMSprop(pre_trained_model.parameters(), lr= args.lr, weight_decay= 5e-6)


print('---------------------Create data loader--------------------')
train_loader = data_utils.generate_data_loader(train_instances, config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=True)
valid_loader = data_utils.generate_data_loader(validate_instances, config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)
test_loader = data_utils.generate_data_loader(test_instances, config_param['batch_size'], item_dict, MAX_SEQ_LENGTH, is_bseq=True, is_shuffle=False)

rec_sys_model, optimizer, cur_epoch, val_loss_min, best_recall, train_losses, train_recalls, val_losses, val_recalls\
    = check_point.load_ckpt(ckpt_path, pre_trained_model, optimizer)
loss_func = loss.Weighted_BCE_Loss()

top_k = config_param['top_k']
train_display_step = 300
val_display_step = 60
test_display_step = 10
epoch = args.epoch

loss_min = val_loss_min
recall_max = best_recall
epsilon = args.epsilon

# train_losses = []
# val_losses = []
# train_recalls = []
# val_recalls = []
# test_losses = []
# test_recalls = []

print('-------------------Continue Training Model---------------------')

############################ Train Model #############################

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

for ep in range(cur_epoch+1, epoch):

    rec_sys_model, optimizer, avg_train_loss, avg_train_recall = model_utils.train_model(rec_sys_model, loss_func, optimizer, train_loader,
                                                                                         ep, top_k, train_display_step)
    train_losses.append(avg_train_loss)
    train_recalls.append(avg_train_recall)

    avg_val_loss, avg_val_recall = model_utils.validate_model(rec_sys_model, loss_func, valid_loader,
                                                              ep, top_k, val_display_step)
    val_losses.append(avg_val_loss)
    val_recalls.append(avg_val_recall)

    # avg_test_loss, avg_test_recall = model_utils.test_model(rec_sys_model, loss_func, test_loader,
    #                                                         ep + 1, top_k, test_display_step)
    # test_losses.append(avg_test_loss)
    # test_recalls.append(avg_test_recall)

    # scheduler.step()

    checkpoint = {
        'epoch': ep,
        'valid_loss_min': avg_val_loss,
        'best_recall': avg_val_recall,
        'state_dict': rec_sys_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_loss_list': val_losses,
        'val_recall_list': val_recalls,
        'train_loss_list': train_losses,
        'train_recall_list': train_recalls
    }
    # save checkpoint
    check_point.save_ckpt(checkpoint, False, model_name, ckpt_dir, best_ckpt_dir, ep)
    check_point.save_config_param(ckpt_dir, model_name, config_param)

    if ((loss_min - avg_val_loss) / loss_min > epsilon and avg_val_recall > recall_max):
        print('Test loss decrease from ({:.5f} --> {:.5f}) '.format(loss_min, avg_val_loss))
        print('Can save model')
        check_point.save_ckpt(checkpoint, True, model_name, ckpt_dir, best_ckpt_dir, ep)
        check_point.save_config_param(best_ckpt_dir, model_name, config_param)
        loss_min = avg_val_loss
        recall_max = avg_val_recall

    print('-' * 100)
    utils.plot_loss(train_losses, val_losses)
    utils.plot_recall(train_recalls, val_recalls)
