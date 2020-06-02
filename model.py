import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

###################  Multi Intent Next Basket Rec Model ####################

class RecSysModel(torch.nn.Module):
    """
      Input data: [u_1, u_2, u_3, u_4, u_5, u_i..., u_n]
      u_i: ith user's seq
      u_i = [b_1, b_2, b_3, b_4, b_j...b_n]
      b_j: jth user's basket which contains items: i_1, i_2, i_3, ....i_n.
    """

    def __init__(self, config_param, max_seq_length, item_probs, adj_matrix, device, d_type):
        super(RecSysModel, self).__init__()
        self.rnn_units = config_param['rnn_units']
        self.rnn_layers = config_param['rnn_layers']
        self.embedding_dim = config_param['embedding_dim']
        self.max_seq_length = max_seq_length
        self.nb_items = len(item_probs)
        self.item_probs = item_probs
        self.alpha = config_param['alpha']
        self.batch_size = config_param['batch_size']
        self.top_k = config_param['top_k']
        self.embed_transformer = config_param['embed_transformer']
        self.num_heads = config_param['num_heads']
        self.n_layers = config_param['n_transformer_layers']
        self.dropout = config_param['dropout']
        self.device = device
        self.d_type = d_type

        # initialized adjacency matrix
        self.A = adj_matrix
        self.A = torch.from_numpy(self.A).to(self.device, d_type)

        # threshold ignore weak correlation
        threshold = adj_matrix.mean()
        print(threshold)
        item_bias = torch.ones(self.nb_items) / self.nb_items
        print(item_bias)
        self.mask = None

        # network architecture
        self.drop_out_1 = nn.Dropout(p=self.dropout)
        self.drop_out_2 = nn.Dropout(p=self.dropout)
        # self.norm1d = nn.LayerNorm(self.embedding_dim)
        self.fc_basket_encoder = nn.Linear(in_features=self.nb_items, out_features=self.embedding_dim, bias=True)
        self.fc_basket_encoder_2 = nn.Linear(in_features=self.nb_items, out_features=self.embedding_dim, bias=True)

        # encoder_layers = TransformerEncoderLayer(d_model = self.embedding_dim, nhead = self.num_heads, dim_feedforward = self.embed_transformer, dropout= self.dropout)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, self.n_layers)
        # self.layer_norm = nn.LayerNorm([self.max_seq_length, self.embedding_dim])
        # self.src_mask = None
        self.seq_encoder = nn.LSTM(self.embedding_dim, self.rnn_units, self.rnn_layers, bias=True, batch_first=True,
                                   dropout=self.dropout)
        # self.W_hidden = nn.Parameter(data=torch.randn(self.rnn_units, self.nb_items).type(self.d_type))
        self.h2item_score = nn.Linear(in_features=self.rnn_units, out_features=self.nb_items, bias=False)
        # self.threshold = nn.Parameter(data=torch.Tensor([threshold]).type(d_type))
        self.I_B = nn.Parameter(data=item_bias.type(d_type))
        # self.init_weight()

    def init_weight(self):
        torch.nn.init.kaiming_uniform_(self.fc_basket_encoder.weight.data, nonlinearity='relu')
        # self.fc_basket_encoder.weight.data.zero_()
        self.fc_basket_encoder.bias.data.zero_()

        torch.nn.init.kaiming_uniform_(self.fc_basket_encoder_2.weight.data, nonlinearity='relu')
        # self.fc_basket_encoder_2.weight.data.zero_()
        self.fc_basket_encoder_2.bias.data.zero_()
        # print(self.fc_basket_encoder.weight.data)

        torch.nn.init.xavier_uniform_(self.h2item_score.weight.data)
        # self.h2item_score.bias.data.zero_()
        # print(self.h2item_score.weight.data)

        for name, param in self.seq_encoder.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.rnn_layers, batch_size, self.rnn_units).to(self.device, self.d_type),
                torch.zeros(self.rnn_layers, batch_size, self.rnn_units).to(self.device, self.d_type))

    def forward(self, x, seq_len):
        # print(seq_len)
        # print(x.size())
        batch_size = x.size()[0]
        item_bias_diag = F.relu(torch.diag(self.I_B))
        x = x.contiguous()
        basket_encoder_1 = self.drop_out_1(F.relu(self.fc_basket_encoder(x)))
        # print(basket_encoder_1)
        basket_encoder_2 = self.drop_out_2(F.relu(self.fc_basket_encoder_2(x)))

        # basket_encoder = (basket_encoder_1 + basket_encoder_2)/2
        combine = torch.cat((basket_encoder_1.unsqueeze(dim=-1), basket_encoder_2.unsqueeze(dim=-1)), dim=-1)
        basket_encoder = torch.max(combine, dim=-1).values

        # print(seq_len)
        # transformer_encoder = self.transformer_encoder(basket_encoder_1.transpose(0,1), src_key_padding_mask= self.create_src_key_padding_mask(seq_len))
        # pack seq for rnn
        # packed_u_seqs = torch.nn.utils.rnn.pack_padded_sequence(basket_encoder, seq_len, batch_first=True,
        #                                                         enforce_sorted=False)
        (h_0, c_0) = self.init_hidden(batch_size)

        # next basket sequence encoder
        lstm_out, (h_n, c_n) = self.seq_encoder(basket_encoder, (h_0, c_0))
        # print(lstm_out)
        actual_index = torch.arange(0, batch_size) * self.max_seq_length + (seq_len - 1)
        actual_lstm_out = lstm_out.reshape(-1, self.rnn_units)[actual_index]

        # print("similar between output hidden lstm")
        # print_similar_list(similar_in_batch(h_n[-1].detach()))
        hidden_to_score = self.h2item_score(actual_lstm_out)
        # print(hidden_to_score)

        # predict next items score
        next_item_probs = torch.sigmoid(hidden_to_score)

        # print(next_item_probs)
        predict = (1 - self.alpha) * next_item_probs + self.alpha * (
                torch.mm(next_item_probs, item_bias_diag) + torch.mm(next_item_probs, self.A))
        # predict score
        # print("similar between predict score")
        # print(similar_in_batch(predict.detach()))
        return predict

    def create_src_key_padding_mask(self, seq_length):
        src_key_padding = torch.ones(seq_length.size()[0], self.max_seq_length)
        for i in range(seq_length.size()[0]):
            src_key_padding[i, : seq_length[i]] = 0
        return src_key_padding.to(self.device).bool()
