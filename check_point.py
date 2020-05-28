import torch
import os
import json
import shutil

######## SAVE AND LOAD CHECKPOINT ##########

def save_ckpt(state, is_best, prefix_name, checkpoint_dir, best_model_dir, epoch):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    parent_folder = os.path.join(checkpoint_dir, prefix_name)
    try:
        os.makedirs(parent_folder, exist_ok=True)
        print("Directory '%s' created successfully" % parent_folder)
    except OSError as error:
        print("Directory '%s' can not be created" % parent_folder)

    ckpt_sub_folder = 'epoch_' + str(epoch)
    path = os.path.join(parent_folder, ckpt_sub_folder)
    try:
        os.makedirs(path, exist_ok=True)
        print("Directory '%s' created successfully" % ckpt_sub_folder)
    except OSError as error:
        print("Directory '%s' can not be created" % ckpt_sub_folder)

    f_path = path + '/' + prefix_name + '_checkpoint.pt'
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_ckpt = os.path.join(best_model_dir, prefix_name)
        try:
            os.makedirs(best_ckpt, exist_ok=True)
            print("Directory '%s' created successfully" % best_ckpt)
        except OSError as error:
            print("Directory '%s' can not be created" % best_ckpt)
        best_fpath = best_ckpt + '/' + prefix_name + '_best_model.pt'
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckpt(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min


def save_config_param(model_dir, prefix_name, config_param):
    config_file = model_dir + prefix_name + '/' + prefix_name + '_config.json'
    with open(config_file, 'w') as fp:
        json.dump(config_param, fp)


def load_config_param(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
        return data


# def save_log_result(log_result_file, train_result, val_result, test_result):
#     with pd.ExcelWriter(log_result_file, mode='w') as writer:
#         train_result.to_excel(writer, sheet_name='train_result_sheet')
#         val_result.to_excel(writer, sheet_name='val_result_sheet')
#         test_result.to_excel(writer, sheet_name='test_result_sheet')