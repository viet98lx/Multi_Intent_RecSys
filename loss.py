import torch
################## Loss Funtion and compute recall fucntion #####################

class Weighted_BCE_Loss(torch.nn.Module):

    def __init__(self, reduction ='mean'):
        super(Weighted_BCE_Loss, self).__init__()
        self.reduction = reduction

    def forward(self, predict, y, pos_weight=None):
        sigmoid_predict = torch.sigmoid(predict)
        # sigmoid_predict = predict
        neg_y = (1.0 - y)
        pos_predict = y * predict

        pos_max = torch.max(pos_predict, dim=1).values.unsqueeze(1)
        # print(pos_max)
        pos_min = torch.min(pos_predict + neg_y * pos_max, dim=1).values.unsqueeze(1)
        # print(pos_min)

        nb_pos = (neg_y == 0).sum(dim=-1).to(predict.dtype)
        nb_neg = (y.size()[1] - nb_pos).to(predict.dtype)

        ratio = (nb_neg / nb_pos).unsqueeze(dim=-1)
        # print(ratio)

        neg_loss = -neg_y * torch.log(1.0 - torch.sigmoid(predict - pos_min))
        # neg_loss = -neg_y * torch.log(1.0 - sigmoid_predict)
        pos_loss = -y * torch.log(sigmoid_predict)

        loss_batch = pos_loss * ratio + neg_loss + 1e-8

        if (self.reduction == 'mean'):
            return loss_batch.mean()
        else:
            return loss_batch.sum()
