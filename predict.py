import os
import torch
import utils

def generate_predict(model, data_loader, result_file, reversed_item_dict, number_predict):
    device = model.device
    nb_test_batch = len(data_loader.dataset) // model.batch_size
    if len(data_loader.dataset) % model.batch_size == 0:
        total_batch = nb_test_batch
    else :
        total_batch = nb_test_batch + 1
    print(total_batch)
    model.eval()
    with torch.no_grad() :
        with open(result_file, 'w') as f:
            f.write('Predict result: ')
            for i, data_pack in enumerate(data_loader,0):
                data_x, data_seq_len, data_y = data_pack
                x_ = data_x.to_dense().to(dtype = model.d_type, device = device)
                real_batch_size = x_.size()[0]
                y_ = data_y.to(dtype = model.d_type, device = device)
                predict_ = model(x_, data_seq_len)
                sigmoid_pred = torch.sigmoid(predict_)
                topk_result = sigmoid_pred.topk(dim=-1, k= number_predict, sorted=True)
                indices = topk_result.indices
                # print(indices)
                values = topk_result.values

                for row in range(0, indices.size()[0]):
                    f.write('\n')
                    f.write('ground truth: ')
                    ground_truth = y_[row].nonzero().squeeze(dim=-1)
                    for idx_key in range(0, ground_truth.size()[0]):
                        f.write(str(reversed_item_dict[ground_truth[idx_key].item()]) + " ")
                    f.write('\n')
                    f.write('predicted items: ')
                    for col in range(0, indices.size()[1]):
                        f.write('| ' + str(reversed_item_dict[indices[row][col].item()]) + ': %.3f' % (values[row][col].item()) + ' ')

def top_k_recall_on_all_dataset(model, data_loader, topK):
    device = model.device
    nb_batch = len(data_loader.dataset) // model.batch_size
    if len(data_loader.dataset) % model.batch_size == 0:
        total_batch = nb_batch
    else :
        total_batch = nb_batch + 1
    print(total_batch)
    list_correct_predict = []
    list_actual_size = []

    model.eval()
    with torch.no_grad() :
        for idx, data_pack in enumerate(data_loader,0):
            x_, data_seq_len, y_ = data_pack
            x_test = x_.to_dense().to(dtype = model.d_type, device = device)
            real_batch_size = x_test.size()[0]
            y_test = y_.to(device = device, dtype = model.d_type)

            logits_predict = model(x_test, data_seq_len)

            predict_basket = utils.predict_top_k(logits_predict, topK, real_batch_size, model.device, model.nb_items)
            correct_predict = predict_basket * y_test
            nb_correct = (correct_predict != 0.0).sum(dim = -1)
            actual_basket_size = (y_test != 0.0).sum(dim = -1)
            for i in range(0, real_batch_size):
                list_correct_predict.append(nb_correct[i].item())
                list_actual_size.append(actual_basket_size[i].item())