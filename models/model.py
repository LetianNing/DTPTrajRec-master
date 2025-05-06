import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from copy import deepcopy
from torch.nn.functional import softplus
from torch.distributions import Normal, Independent
import torch
from torch.distributions import Normal, kl_divergence


def mask_log_softmax(x, mask, log_flag=True):
    maxes = torch.max(x, 1, keepdim=True)[0] + 1e-5
    x_exp = torch.exp(x - maxes) * mask
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    if log_flag:
        output_custom = torch.log(x_exp / x_exp_sum)
    else:
        output_custom = x_exp / x_exp_sum
    return output_custom




class GaussianEstimator1(nn.Module):
    def __init__(self, args):
        super(GaussianEstimator1, self).__init__()

        self.z_dim = args.z_dim

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(args.hid_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.z_dim * 2),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        params = self.net(x)

        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive
        return Independent(Normal(loc=mu, scale=sigma), 1)




class Encoder1(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim

        input_dim = parameters.id_emb_dim * 2
        self.zoneout_prob = 0.5
        self.rnn = nn.GRU(input_dim, self.hid_dim)

    def forward(self, src, src_len, pro_features):

        packed_embedded = nn.utils.rnn.pack_padded_sequence(src, src_len, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embedded)


        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        return outputs, hidden


class Attention(nn.Module):
    # TODO update to more advanced attention layer.
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim

        self.v = nn.Linear(self.hid_dim, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(2 * self.hid_dim, self.hid_dim),
            nn.Sigmoid()
        )

    def forward(self, hidden, encoder_outputs, attn_mask):
        # hidden = [1, bath size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * num directions]
        src_len = encoder_outputs.shape[0]
        # repeat decoder hidden sate src_len times
        hidden = hidden.repeat(src_len, 1, 1)
        hidden = hidden.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src len, hid dim]
        # encoder_outputs = [batch size, src len, hid dim * num directions]
        concatenated = torch.cat((hidden, encoder_outputs), dim=2)
        g = self.gate(concatenated)
        energy = torch.tanh(g * hidden + (1 - g) * encoder_outputs)
        # energy = [batch size, src len, hid dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]
        attention = attention.masked_fill(attn_mask == 0, -1e10)
        # using mask to force the attention to only be over non-padding elements.

        return F.softmax(attention, dim=1)


class DecoderMulti(nn.Module):
    def __init__(self, parameters, SE, spatial_A_trans):
        super().__init__()

        self.id_size = parameters.id_size
        self.id_emb_dim = parameters.id_emb_dim
        self.hid_dim = parameters.hid_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_dim = parameters.online_dim
        self.rid_fea_dim = parameters.rid_fea_dim

        self.attn_flag = parameters.attn_flag
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag  # final softmax
        self.online_features_flag = parameters.online_features_flag
        self.tandem_fea_flag = parameters.tandem_fea_flag
        self.transition_matrix = nn.Parameter(spatial_A_trans)
        self.emb_id = nn.Embedding(self.id_size, self.id_emb_dim)
        rnn_input_dim = self.id_emb_dim + 1
        fc_id_out_input_dim = self.hid_dim
        fc_rate_out_input_dim = self.hid_dim

        type_input_dim = self.id_emb_dim + self.hid_dim
        self.tandem_fc = nn.Sequential(
            nn.Linear(type_input_dim, self.hid_dim),
            nn.ReLU()
        )

        if self.attn_flag:
            self.attn = Attention(parameters)
            rnn_input_dim = rnn_input_dim + self.hid_dim

        if self.online_features_flag:
            rnn_input_dim = rnn_input_dim + self.online_dim  # 5 poi and 5 road network

        if self.tandem_fea_flag:
            fc_rate_out_input_dim = self.hid_dim + self.rid_fea_dim

        self.rnn = nn.GRU(rnn_input_dim, self.hid_dim)
        self.fc_id_out = nn.Linear(fc_id_out_input_dim, self.id_size)
        self.fc_rate_out = nn.Linear(fc_rate_out_input_dim, 1)
        self.dropout = nn.Dropout(parameters.dropout)
        self.gate = nn.Sequential(
            nn.Linear(self.id_size, self.id_size),
            nn.Sigmoid()
        )

    def mask_trg_mask(self, N, trg_len):
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        # trg_mask shape:(N, 1, 1, trg_len)
        return trg_mask.to(self.device)

    def forward(self, input_id, input_rate, hidden, encoder_outputs, attn_mask,
                pre_grid, next_grid, constraint_vec, pro_features, online_features, rid_features):

        # input_id = [batch size, 1] rid long
        # input_rate = [batch size, 1] rate float.
        # hidden = [1, batch size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * num directions]
        # attn_mask = [batch size, src len]
        # pre_grid = [batch size, 3]
        # next_grid = [batch size, 3]
        # constraint_vec = [batch size, id_size], [id_size] is the vector of reachable rid
        # pro_features = [batch size, profile features input dim]
        # online_features = [batch size, online features dim]
        # rid_features = [batch size, rid features dim]

        input_id = input_id.squeeze(1).unsqueeze(0)  # cannot use squeeze() bug for batch size = 1
        # input_id = [1, batch size]
        input_rate = input_rate.unsqueeze(0)
        # input_rate = [1, batch size, 1]
        embedded = self.dropout(self.emb_id(input_id))
        # embedded = [1, batch size, emb dim]

        # rnn_input = torch.cat((embedded, input_rate), dim=2)

        if self.attn_flag:
            a = self.attn(hidden, encoder_outputs, attn_mask)
            # a = [batch size, src len]
            a = a.unsqueeze(1)
            # a = [batch size, 1, src len]
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            # encoder_outputs = [batch size, src len, hid dim * num directions]
            weighted = torch.bmm(a, encoder_outputs)
            # weighted = [batch size, 1, hid dim * num directions]
            weighted = weighted.permute(1, 0, 2)
            # weighted = [1, batch size, hid dim * num directions]

            if self.online_features_flag:
                rnn_input = torch.cat((weighted, embedded, input_rate,
                                       online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((weighted, embedded, input_rate), dim=2)
        else:
            if self.online_features_flag:
                rnn_input = torch.cat((embedded, input_rate, online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((embedded, input_rate), dim=2)

        output, hidden = self.rnn(rnn_input, hidden)

        transition_matrix = self.gate(self.transition_matrix)

        # pre_rid
        if self.dis_prob_mask_flag:
            prediction_id =mask_log_softmax(torch.matmul(self.fc_id_out(output.squeeze(0)), transition_matrix),
                             constraint_vec, log_flag=True)
        else:
            prediction_id = F.log_softmax(torch.matmul(self.fc_id_out(output.squeeze(0)), transition_matrix), dim=1)
            # then the loss function should change to nll_loss()

        # pre_rate
        max_id = prediction_id.argmax(dim=1).long()
        id_emb = self.dropout(self.emb_id(max_id))
        rate_input = torch.cat((id_emb, hidden.squeeze(0)), dim=1)
        rate_input = self.tandem_fc(rate_input)  # [batch size, hid dim]
        if self.tandem_fea_flag:
            prediction_rate = torch.sigmoid(self.fc_rate_out(torch.cat((rate_input, rid_features), dim=1)))
        else:
            prediction_rate = torch.sigmoid(self.fc_rate_out(rate_input))

        # prediction_id = [batch size, id_size]
        # prediction_rate = [batch size, 1]

        return prediction_id, prediction_rate, hidden


class DTPTrajRec(nn.Module):
    def __init__(self, encoder1, sample1, decoder, parameters, SE, base_channel,
                 x_id, y_id):
        super(DTPTrajRec, self).__init__()
        self.encoder1 = encoder1  # Encoder

        self.sample1 = sample1

        self.decoder = decoder

        self.device = "cuda:0"
        self.id_size = parameters.id_size

        self.linear_ronhe_hidden = nn.Linear(parameters.z_dim * 2, parameters.hid_dim)



        self.linear_hidden = nn.Linear(parameters.z_dim, parameters.hid_dim)

        self.encoder_out = nn.Sequential(
            nn.Linear(512 + 64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )

        self.Embed1 = nn.Embedding(x_id + 1, parameters.id_emb_dim)
        self.Embed2 = nn.Embedding(y_id + 1, parameters.id_emb_dim)
        # self.Embed3 = nn.Embedding(parameters.id_size, 32)

    def fill_new_tensor_blank(self, original_grid, original_gps, max_len):

        new__grid = torch.zeros((original_grid.shape[0], max_len, original_grid.shape[2]),
                                device=original_grid.device)

        new_gps = torch.zeros((original_grid.shape[0], max_len, original_grid.shape[2]),
                              device=original_grid.device)

        for i in range(original_grid.shape[0]):
            for j in range(original_grid.shape[1]):
                x, y, t = original_grid[i, j].cpu().numpy()
                gps_x, gps_y = original_gps[i, j].cpu().numpy()
                t_index = int(t)
                if 0 <= t_index < max_len:
                    new__grid[i, t_index] = torch.tensor([x, y, t], device=original_grid.device)
                    new_gps[i, t_index] = torch.tensor([gps_x, gps_y, t], device=original_gps.device)

        return new__grid, new_gps


    def compute_similarity(self, dist1, dist2, sigma=None):

        mu1, sigma1 = dist1.base_dist.loc, dist1.base_dist.scale
        mu2, sigma2 = dist2.base_dist.loc, dist2.base_dist.scale

        mean_diff = torch.abs(mu1 - mu2)
        scale_diff = torch.abs(sigma1 - sigma2)
        wasserstein_dist = mean_diff + scale_diff

        if sigma is None:
            sigma = 1.0

        similarity = torch.exp(-wasserstein_dist / sigma)

        return similarity

    def forward(self, src_gps_seqs, user_tf_idf, spatial_A_trans, src_road_index_seqs, SE, tra_time_A,
                tra_loca_A,
                src_len, src_grid_seqs, src_eid_seqs, src_rate_seqs, trg_in_grid_seqs,
                trg_in_index_seqs, trg_rids, trg_rates, trg_len,
                pre_grids, next_grids, constraint_mat, pro_features,
                online_features_dict, rid_features_dict,
                teacher_forcing_ratio=0.5):

        batchsize, _, _ = src_grid_seqs.shape
        max_trg_len = trg_rids.size(0)
        attn_mask = torch.zeros(batchsize, max(trg_len))  # only attend on unpadded sequence
        for i in range(len(trg_len)):
            attn_mask[i][:trg_len[i]] = 1.
        attn_mask = attn_mask.to(self.device)

        src_grid_seqs2, _ = self.fill_new_tensor_blank(src_grid_seqs, src_gps_seqs, max_trg_len)
        pre_grids = pre_grids.permute(1, 0, 2)
        next_grids = next_grids.permute(1, 0, 2)

        src_grid_seqs2_x = self.Embed1(src_grid_seqs2[:, :, 0].long())
        pre_grids_x = self.Embed1(pre_grids[:, :, 0].long())
        next_grids_x = self.Embed1(next_grids[:, :, 0].long())

        src_grid_seqs2_y = self.Embed2(src_grid_seqs2[:, :, 1].long())
        pre_grids_y = self.Embed2(pre_grids[:, :, 1].long())
        next_grids_y = self.Embed2(next_grids[:, :, 1].long())

        src_grid_seqs2 = torch.cat((src_grid_seqs2_x, src_grid_seqs2_y), dim=-1)
        pre_grids = torch.cat((pre_grids_x, pre_grids_y), dim=-1)
        next_grids = torch.cat((next_grids_x, next_grids_y), dim=-1)



        pre_grids = pre_grids.permute(1, 0, 2)
        next_grids = next_grids.permute(1, 0, 2)


        src_attention, src_hidden = self.encoder1(src_grid_seqs2.permute(1, 0, 2), trg_len, pro_features)
        p_cur = self.sample1(src_hidden.squeeze(0))

        src_attention_pred, src_hidden2_pred = self.encoder1(pre_grids, trg_len, pro_features)
        p_pre = self.sample1(src_hidden2_pred.squeeze(0))
        z_pre = p_pre.rsample()

        src_attention_next, src_hidden2_next = self.encoder1(next_grids, trg_len, pro_features)
        p_next = self.sample1(src_hidden2_next.squeeze(0))
        z_next = p_next.rsample()

        kl_cur_pre = self.compute_similarity(p_cur, p_pre)
        kl_cur_next = self.compute_similarity(p_cur, p_next)

        src_hidden_sample = self.linear_ronhe_hidden(torch.cat((z_pre * kl_cur_pre, z_next * kl_cur_next), dim=-1))
        src_hidden = src_hidden + src_hidden_sample

        src_attention = src_attention + src_attention_pred + src_attention_next

        all_road_embed = torch.einsum('btr,rd->btd', (constraint_mat.permute(1, 0, 2), SE))  # B, T, F
        summ = constraint_mat.permute(1, 0, 2).sum(-1).unsqueeze(-1)
        trajectory_point_embed = all_road_embed / summ

        trajectory_point_sum = trg_in_index_seqs.sum(1)
        trajectory_embed = (trajectory_point_embed.sum(1) / trajectory_point_sum).unsqueeze(0)
        src_hidden = self.encoder_out(torch.cat((src_hidden, trajectory_embed), -1))

        outputs_id, outputs_rate = self.normal_step(max_trg_len, batchsize, trg_rids, trg_rates, trg_len,
                                                    src_attention, src_hidden, attn_mask,
                                                    online_features_dict,
                                                    rid_features_dict,
                                                    pre_grids, next_grids, constraint_mat, pro_features,
                                                    teacher_forcing_ratio)

        return outputs_id, outputs_rate

    def normal_step(self, max_trg_len, batch_size, trg_id, trg_rate, trg_len, encoder_outputs, hidden,
                    attn_mask, online_features_dict, rid_features_dict,
                    pre_grids, next_grids, constraint_mat, pro_features, teacher_forcing_ratio):
        """
        Returns:
        -------
        outputs_id: [seq len, batch size, id size]
        outputs_rate: [seq len, batch size, 1]
        """
        # tensor to store decoder outputs
        outputs_id = torch.zeros(max_trg_len, batch_size, self.decoder.id_size).to(self.device)
        outputs_rate = torch.zeros(trg_rate.size()).to(self.device)

        # first input to the decoder is the <sos> tokens 0
        input_id = trg_id[0, :]
        input_rate = trg_rate[0, :]
        for t in range(1, max_trg_len):
            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and attn_mask
            # receive output tensor (predictions) and new hidden state
            if self.decoder.online_features_flag:
                online_features = get_dict_info_batch(input_id, online_features_dict).to(self.device)
            else:
                online_features = torch.zeros((1, batch_size, self.decoder.online_dim))
            if self.decoder.tandem_fea_flag:
                rid_features = get_dict_info_batch(input_id, rid_features_dict).to(self.device)
            else:
                rid_features = None
            prediction_id, prediction_rate, hidden = self.decoder(input_id, input_rate, hidden, encoder_outputs,
                                                                  attn_mask, pre_grids[t], next_grids[t],
                                                                  constraint_mat[t], pro_features, online_features,
                                                                  rid_features)

            # place predictions in a tensor holding predictions for each token
            outputs_id[t] = prediction_id
            outputs_rate[t] = prediction_rate

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1_id = prediction_id.argmax(1)
            top1_id = top1_id.unsqueeze(-1)  # make sure the output has the same dimension as input

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input_id = trg_id[t] if teacher_force else top1_id
            input_rate = trg_rate[t] if teacher_force else prediction_rate

        # max_trg_len, batch_size, trg_rid_size
        outputs_id = outputs_id.permute(1, 0, 2)  # batch size, seq len, rid size
        outputs_rate = outputs_rate.permute(1, 0, 2)  # batch size, seq len, 1
        for i in range(batch_size):
            outputs_id[i][trg_len[i]:] = 0
            outputs_id[i][trg_len[i]:, 0] = 1
            outputs_rate[i][trg_len[i]:] = 0
        outputs_id = outputs_id.permute(1, 0, 2)
        outputs_rate = outputs_rate.permute(1, 0, 2)

        return outputs_id, outputs_rate