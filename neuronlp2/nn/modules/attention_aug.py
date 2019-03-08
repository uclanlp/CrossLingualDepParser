#

# Attention Layer with augmented feature inputs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from .sparse import Embedding

# including the use of both-side POS and pair-dist features
# digits are arranged: (dist, e-pos, d-pos)
class AugFeatureHelper(nn.Module):
    def __init__(self, max_dist, use_neg_dist, num_pos, use_encoder_pos, use_decoder_pos):
        super(AugFeatureHelper, self).__init__()
        #
        self.max_dist = max_dist
        self.use_neg_dist = use_neg_dist
        self.num_pos = num_pos
        self.use_encoder_pos = use_encoder_pos
        self.use_decoder_pos = use_decoder_pos
        # 0: dist, 1: e-pos, 2: d-pos
        self.num_features = 2*self.max_dist+1 if self.use_neg_dist else self.max_dist+1
        if use_encoder_pos:
            self.alpha_epos = self.num_features
            self.num_features *= num_pos
        else:
            self.alpha_epos = 0
        if use_decoder_pos:
            self.alpha_dpos = self.num_features
            self.num_features *= num_pos
        else:
            self.alpha_dpos = 0

    def get_num_features(self):
        return self.num_features

    # [batch, len-d, len-e], [batch, len-e], [batch, len-d]
    # return idxes of [batch, len-d, len-e]
    def get_final_features(self, raw_dists, encoder_pos, decoder_pos):
        if not self.use_neg_dist:
            raw_dists = torch.abs(raw_dists)
        raw_dists = torch.clamp(raw_dists, min=-self.max_dist, max=self.max_dist)
        encoder_pos = encoder_pos.unsqueeze(1)
        decoder_pos = decoder_pos.unsqueeze(2)
        #
        output = raw_dists
        if self.alpha_epos > 0:
            output = output + self.alpha_epos * encoder_pos
        if self.alpha_dpos > 0:
            output = output + self.alpha_dpos * decoder_pos
        return output

class AugBiAAttention(nn.Module):
    def __init__(self, input_size_encoder, input_size_decoder, num_labels, num_features, dim_feature, drop_f_embed, biaffine=True):
        super(AugBiAAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.dim_feature = dim_feature
        self.biaffine = biaffine
        # original parameters
        self.W_d = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder))
        self.W_e = Parameter(torch.Tensor(self.num_labels, self.input_size_encoder))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder, self.input_size_encoder))
        else:
            self.register_parameter('U', None)
        # extra param for aug-features
        self.use_features = (num_features > 1)
        self.E_drop = nn.Dropout(p=drop_f_embed)
        if self.use_features:        # only meaningful if >1 features
            self.E = Embedding(num_features, dim_feature)
            # concated to the enc-side, thus map to dec-size (same for all num_labels)
            self.U_f = Parameter(torch.Tensor(dim_feature, self.input_size_decoder))
            self.W_f = Parameter(torch.Tensor(dim_feature, self.num_labels,))
        else:
            self.add_module('E', None)
            self.register_parameter('U_f', None)
            self.register_parameter('W_f', None)
        #
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.W_d)
        nn.init.xavier_uniform(self.W_e)
        nn.init.constant(self.b, 0.)
        if self.biaffine:
            nn.init.xavier_uniform(self.U)
        if self.use_features:
            nn.init.xavier_uniform(self.U_f)
            nn.init.xavier_uniform(self.W_f)

    """
    input_d: [batch, len-d, size-d], input_e: [batch, len-e, size-e]
    input_features: ints of [batch, len-d, len-e]
    """
    def forward(self, input_d, input_e, input_features, mask_d=None, mask_e=None):
        assert input_d.size(0) == input_e.size(0), 'batch sizes of encoder and decoder are requires to be equal.'
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()

        # compute decoder part: [num_label, input_size_decoder] * [batch, input_size_decoder, length_decoder]
        # the output shape is [batch, num_label, length_decoder]
        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)
        # compute decoder part: [num_label, input_size_encoder] * [batch, input_size_encoder, length_encoder]
        # the output shape is [batch, num_label, length_encoder]
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)

        # scores with features
        if self.use_features:
            features_embed = self.E_drop(self.E(Variable(input_features)))     # [batch, len-d, len-e, fdim]
            features_out0 = torch.matmul(features_embed, self.W_f)    # [batch, len-d, len-e, num_label]
            output_f = features_out0.transpose(2, 3).transpose(1, 2)    # [batch, num_label, len-d, len-e]

        # output shape [batch, num_label, length_decoder, length_encoder]
        if self.biaffine:
            # compute bi-affine part
            # [batch, 1, length_decoder, input_size_decoder] * [num_labels, input_size_decoder, input_size_encoder]
            # output shape [batch, num_label, length_decoder, input_size_encoder]
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            # [batch, num_label, length_decoder, input_size_encoder] * [batch, 1, input_size_encoder, length_encoder]
            # output shape [batch, num_label, length_decoder, length_encoder]
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))

            if self.use_features:
                # [batch, len-d, len-e, input_size_encoder] .dot [batch, 1, len-e, input_size_encoder]
                # -> [batch, 1, len-d, len-e]
                # output_f2 = torch.sum(features_embed*input_e.unsqueeze(1), dim=-1).unsqueeze(1)
                #
                # [batch, len-e, len-d, *] * [batch, len-e, *, 1]
                # output_f2 = torch.matmul(features_embed.transpose(1,2), input_e.unsqueeze(-1)).transpose(1,2).squeeze(-1).unsqueeze(1)

                # [batch, len-d, len-e, input_size_decoder]
                features_embed_map = torch.matmul(features_embed, self.U_f)
                # [batch, len-d, len-e, *] * [batch, len-d, *, 1]
                output_f2 = torch.matmul(features_embed_map, input_d.unsqueeze(-1)).squeeze(-1).unsqueeze(1)

                output = output + out_d + out_e + output_f + output_f2 + self.b
            else:
                output = output + out_d + out_e + self.b
        else:
            if self.use_features:
                output = out_d + out_e + output_f + self.b
            else:
                output = out_d + out_e + self.b

        if mask_d is not None:
            output = output * mask_d.unsqueeze(1).unsqueeze(3) * mask_e.unsqueeze(1).unsqueeze(2)

        return output
