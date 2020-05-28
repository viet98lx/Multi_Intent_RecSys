import torch
################## Loss Funtion and compute recall fucntion #####################

class Weighted_BCE_Loss(torch.nn.Module):

    def __init__(self):
        super(Weighted_BCE_Loss, self).__init__()

    def forward(self, predict, y, pos_weight=None, weight=None):
        sigmoid_predict = torch.sigmoid(predict)
        # sigmoid_predict = predict
        neg_y = (1.0 - y)
        pos_predict = y * predict

        pos_max = torch.abs(torch.max(pos_predict, dim=1).values.unsqueeze(1))
        # print(pos_max)
        pos_min = torch.min(pos_predict + neg_y * pos_max, dim=1).values.unsqueeze(1)
        # print(pos_min)

        nb_pos = (neg_y == 0).sum(dim=1).to(predict.dtype)
        nb_neg = (y.size()[1] - nb_pos).to(predict.dtype)
        ratio = (nb_neg / (nb_pos + 1e-6)).unsqueeze(1)

        neg_loss = -neg_y * torch.log(1.0 - torch.sigmoid(predict - pos_min))
        pos_loss = -y * torch.log(sigmoid_predict)

        loss_batch = pos_loss * ratio + neg_loss + 1e-8

        if weight is not None:
            weight.to(dtype=predict.dtype, device=predict.device)
            loss_batch = loss_batch * weight

        return loss_batch.mean()

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