import torch
import torch.nn as nn

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

        # network architecture
        self.drop_out1 = nn.Dropout(p=self.dropout)
        self.drop_out2 = nn.Dropout(p=self.dropout)
        # self.norm1d = nn.LayerNorm(self.embedding_dim)
        self.fc_basket_encoder = nn.Linear(in_features=self.nb_items, out_features=self.embedding_dim, bias=True)
        self.fc_basket_encoder_2 = nn.Linear(in_features=self.nb_items, out_features=self.embedding_dim, bias=True)

        # encoder_layers = TransformerEncoderLayer(d_model = self.embedding_dim, nhead = self.num_heads, dim_feedforward = self.embed_transformer, dropout= self.dropout)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, self.n_layers)
        # self.layer_norm = nn.LayerNorm([self.max_seq_length, self.embedding_dim])
        # self.src_mask = None
        self.seq_encoder = nn.LSTM(self.embedding_dim, self.rnn_units, self.rnn_layers, bias=True, batch_first=True)
        # self.W_hidden = nn.Parameter(data=torch.randn(self.rnn_units, self.nb_items).type(self.d_type))
        self.h2item_score = nn.Linear(in_features=self.rnn_units, out_features=self.nb_items, bias=False)
        self.threshold = nn.Parameter(data=torch.Tensor([threshold]).type(d_type))
        self.I_B = nn.Parameter(data=item_bias.type(d_type))
        self.init_weight()

    def init_weight(self):
        torch.nn.init.kaiming_uniform_(self.fc_basket_encoder.weight.data)
        # self.fc_basket_encoder.weight.data.zero_()
        self.fc_basket_encoder.bias.data.zero_()

        torch.nn.init.kaiming_uniform_(self.fc_basket_encoder_2.weight.data)
        # self.fc_basket_encoder_2.weight.data.zero_()
        self.fc_basket_encoder_2.bias.data.zero_()
        # print(self.fc_basket_encoder.weight.data)

        torch.nn.init.xavier_uniform_(self.h2item_score.weight.data)
        # self.h2item_score.bias.data.zero_()
        # print(self.h2item_score.weight.data)

        # for name, param in self.seq_encoder.named_parameters():
        #   if 'bias' in name:
        #     nn.init.constant_(param, 0.0)
        #   elif 'weight' in name:
        #     nn.init.xavier_uniform_(param)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.rnn_layers, batch_size, self.rnn_units).to(self.device, self.d_type),
                torch.zeros(self.rnn_layers, batch_size, self.rnn_units).to(self.device, self.d_type))

    def forward(self, x, seq_len):
        # print(seq_len)
        batch_size = x.size()[0]
        # print("similar between one hot basket")
        # print_similar_list(similar_in_batch(x.detach()))

        item_bias_diag = torch.diag(F.relu(self.I_B))
        x = x.contiguous()
        x = x.view(-1, self.nb_items)
        encoder_x = torch.mm(x, item_bias_diag) + F.relu(torch.mm(x, self.A) - torch.abs(self.threshold))

        # encoder_x = encoder_x.view(-1, self.max_seq_length, self.nb_items)
        basket_encoder_1 = self.drop_out1(F.relu(self.fc_basket_encoder(x)))
        basket_encoder_2 = self.drop_out2(F.relu(self.fc_basket_encoder_2(x)))
        combine = torch.cat((basket_encoder_1.unsqueeze(dim=-1), basket_encoder_2.unsqueeze(dim=-1)), dim=-1)
        basket_encoder = torch.max(combine, dim=-1).values
        # print("similar between basket encoder sequence")
        # print_similar_list(similar_in_batch(basket_encoder.detach()))

        # transformer encoder
        # transformer_encoder = self.transformer_encoder(basket_encoder.transpose(0,1), src_key_padding_mask = self.create_src_key_padding_mask(seq_len))

        # pack seq for rnn
        packed_u_seqs = torch.nn.utils.rnn.pack_padded_sequence(basket_encoder, seq_len, batch_first=True,
                                                                enforce_sorted=False)
        (h_0, c_0) = self.init_hidden(batch_size)

        # next basket sequence encoder
        lstm_out, (h_n, c_n) = self.seq_encoder(packed_u_seqs, (h_0, c_0))

        # actual_len = torch.LongTensor(seq_len)
        # masks = (actual_len-1).unsqueeze(0).unsqueeze(2).expand(1, lstm_out.size(1), lstm_out.size(2)).to(self.device)
        # output = lstm_out.gather(0, masks)[0]

        # print("similar between output hidden lstm")
        # print_similar_list(similar_in_batch(h_n[-1].detach()))
        hidden_to_score = self.h2item_score(h_n[-1])

        # predict next items score
        next_item_probs = torch.sigmoid(hidden_to_score)

        # print(next_item_probs)
        predict = (1 - self.alpha) * next_item_probs + self.alpha * (
                    torch.mm(next_item_probs, item_bias_diag) + F.relu(torch.mm(next_item_probs, self.A)))
        # + F.relu(torch.mm(next_item_probs, self.A)))
        # predict score
        # print("similar between predict score")
        # print(similar_in_batch(predict.detach()))
        return predict

    def create_src_key_padding_mask(self, seq_length):
        src_key_padding = torch.ones(len(seq_length), self.max_seq_length)
        for i in range(len(seq_length)):
            src_key_padding[i, : seq_length[i]] = 0
        return src_key_padding.to(self.device).bool()

