import torch
import time
import utils

def train_model(model, loss_func, optimizer, train_loader, epoch, top_k, train_display_step):
    running_train_loss = 0.0
    running_train_recall = 0.0
    device = model.device
    nb_train_batch = len(train_loader.dataset) // model.batch_size
    if len(train_loader.dataset) % model.batch_size == 0:
        total_train_batch = nb_train_batch
    else:
        total_train_batch = nb_train_batch + 1
    model.train()
    start = time.time()

    for i, data in enumerate(train_loader, 0):

        user_seq, train_seq_len, target_basket = data
        x_train_batch = user_seq.to_dense().to(dtype=model.d_type, device=device)
        real_batch_size = x_train_batch.size()[0]
        hidden = model.init_hidden(real_batch_size)
        y_train = target_basket.to(device=device, dtype=model.d_type)

        optimizer.zero_grad()  # clear gradients for this training step

        predict = model(x_train_batch, train_seq_len, hidden)  # predicted output
        loss = loss_func(predict, y_train)  # WBCE loss
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # update gradient

        train_loss_item = loss.item()
        running_train_loss += train_loss_item
        avg_train_loss = running_train_loss / (i + 1)

        train_recall_item = utils.compute_recall_at_top_k(model, predict.clone().detach(), top_k, y_train.clone().detach(), real_batch_size, device)
        running_train_recall += train_recall_item
        avg_train_recall = running_train_recall / (i + 1)

        end = time.time()

        if ((i + 1) % train_display_step == 0 or (i + 1) == total_train_batch):  # print every 50 mini-batches
            top_pred = predict.clone().detach().topk(dim=-1, k=model.top_k, sorted=True)
            print(
                '[Epoch : % d ,Batch Index : %d / %d] Train Loss : %.8f -------- Train Recall@%d: %.8f -------- Time : %.3f seconds ' %
                (epoch, i + 1, total_train_batch, avg_train_loss, top_k, avg_train_recall, end - start))
            print("top k indices predict: ")
            print('--------------------------------------------------------------')
            print('*****  indices *****')
            print(top_pred.indices)
            print('*****  values *****')
            print(top_pred.values)
            print('--------------------------------------------------------------')

            start = time.time()

    print('finish a train epoch')
    return model, optimizer, avg_train_loss, avg_train_recall


def validate_model(model, loss_func, valid_loader, epoch, top_k, val_display_step):
    running_val_loss = 0.0
    running_val_recall = 0.0
    device = model.device
    nb_val_batch = len(valid_loader.dataset) // model.batch_size
    if len(valid_loader.dataset) % model.batch_size == 0:
        total_val_batch = nb_val_batch
    else:
        total_val_batch = nb_val_batch + 1

    model.eval()

    for valid_i, valid_data in enumerate(valid_loader, 0):
        valid_in, valid_seq_len, valid_out = valid_data
        x_valid = valid_in.to_dense().to(dtype=model.d_type, device=device)
        val_batch_size = x_valid.size()[0]
        hidden = model.init_hidden(val_batch_size)
        y_valid = valid_out.to(device=device, dtype=model.d_type)

        valid_predict = model(x_valid, valid_seq_len, hidden)
        val_loss = loss_func(valid_predict, y_valid)

        val_loss_item = val_loss.item()
        running_val_loss += val_loss_item
        avg_val_loss = running_val_loss / (valid_i + 1)

        val_recall_item = utils.compute_recall_at_top_k(model, valid_predict, top_k, y_valid, val_batch_size, device)
        running_val_recall += val_recall_item
        avg_val_recall = running_val_recall / (valid_i + 1)

        if ((valid_i + 1) % val_display_step == 0 or (
                valid_i + 1) == total_val_batch):  # print every 50 mini-batches
            print('[Epoch : % d ,Valid batch Index : %d / %d] Valid Loss : %.8f -------- Valid Recall@%d: %.8f' %
                  (epoch, valid_i + 1, total_val_batch, avg_val_loss, top_k, avg_val_recall))

    return avg_val_loss, avg_val_recall


def test_model(model, loss_func, test_loader, epoch, top_k, test_display_step):
    running_test_recall = 0.0
    running_test_loss = 0.0
    device = model.device
    nb_test_batch = len(test_loader.dataset) // model.batch_size
    if len(test_loader.dataset) % model.batch_size == 0:
        total_test_batch = nb_test_batch
    else:
        total_test_batch = nb_test_batch + 1

    model.eval()

    for test_i, test_data in enumerate(test_loader, 0):
        test_in, test_seq_len, test_out = test_data
        x_test = test_in.to_dense().to(dtype=model.d_type, device=device)
        real_test_batch_size = x_test.size()[0]
        hidden = model.init_hidden(real_test_batch_size)
        y_test = test_out.to(device=device, dtype=model.d_type)

        test_predict = model(x_test, test_seq_len, hidden)
        test_loss = loss_func(test_predict, y_test)

        test_loss_item = test_loss.item()
        running_test_loss += test_loss_item
        avg_test_loss = running_test_loss / (test_i + 1)

        running_test_recall += utils.compute_recall_at_top_k(model, test_predict, top_k, y_test, real_test_batch_size,
                                                       device)
        avg_test_recall = running_test_recall / (test_i + 1)
        if ((test_i + 1) % test_display_step == 0 or (test_i + 1) == total_test_batch):
            print('[Epoch : % d , Test batch_index : %3d --------- Test loss: %.8f ------------- Test Recall@%d : %.8f' %
                  (epoch, test_i + 1, avg_test_loss, top_k, avg_test_recall))
    return avg_test_loss, avg_test_recall