import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F

class Encoder_rnn(nn.Module):
  def __init__(self, args, input_size, hidden_size):
    super(Encoder_rnn, self).__init__()
    self.args = args


    self.rnn = nn.LSTM(input_size = input_size,
              hidden_size = hidden_size,
              num_layers = self.args.num_layers,
              batch_first = True,
              dropout = self.args.dropout,
              bidirectional = True)

  def forward(self, emb, emb_len, all_hiddens=True):

    emb_len = np.array(emb_len)
    sorted_idx= np.argsort(-emb_len)
    emb = emb[sorted_idx]
    emb_len = emb_len[sorted_idx]
    unsorted_idx = np.argsort(sorted_idx)

    packed_emb = torch.nn.utils.rnn.pack_padded_sequence(emb, emb_len, batch_first=True)
    output, hn = self.rnn(packed_emb)

    if all_hiddens:
      unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)
      unpacked = unpacked.transpose(0, 1)
      unpacked = unpacked[torch.LongTensor(unsorted_idx)]
      return unpacked
    else:
      ret = hn[0][-2:].transpose(1,0).contiguous()[torch.LongTensor(unsorted_idx)]
      ret = ret.view(ret.shape[0], -1)
      return ret


'''
From DrQA repo https://github.com/facebookresearch/DrQA
Single head attention
'''
class MatchAttn(nn.Module):
  """Given sequences X and Y, match sequence Y to each element in X.
  * o_i = sum(alpha_j * y_j) for i in X
  * alpha_j = softmax(y_j * x_i)
  """

  def __init__(self, input_size1, input_size2=None, identity=False):
    super(MatchAttn, self).__init__()
    if input_size2 is None:
      input_size2 = input_size1
    hidden_size = min(input_size1, input_size2)
    if not identity:
      self.linear_x = nn.Linear(input_size1, hidden_size)
      self.linear_y = nn.Linear(input_size2, hidden_size)
    else:
      self.linear_x = None
      self.linear_y = None

    self.w = nn.Linear(hidden_size, hidden_size, bias=False)

  def forward(self, x, y, y_mask, is_score=False, no_diag=False):
    """
    Args:
      x: batch * len1 * hdim
      y: batch * len2 * hdim
      y_mask: batch * len2 (1 for padding, 0 for true)
    Output:
      matched_seq: batch * len1 * hdim
    """
    # Project vectors
    if self.linear_x:
      x_proj = self.linear_x(x.view(-1, x.size(2))).view(x.shape[0], x.shape[1], -1)
      x_proj = F.relu(x_proj)
      y_proj = self.linear_y(y.view(-1, y.size(2))).view(y.shape[0], y.shape[1], -1)
      y_proj = F.relu(y_proj)
    else:
      x_proj = x
      y_proj = y

    x_proj = self.w(x_proj)

    # Compute scores

    scores = x_proj.bmm(y_proj.transpose(2, 1))
    # Mask padding

    y_mask = y_mask.unsqueeze(1).expand(scores.size())

    scores.data.masked_fill_(y_mask.bool().data, -float('inf'))
    # Normalize with softmax
    if is_score:
      return scores
    alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=-1)
    alpha = alpha_flat.view(-1, x.size(1), y.size(1))


    # Take weighted average
    matched_seq = alpha.bmm(y)
    #residual_rep = torch.abs(x - matched_seq)

    return matched_seq, alpha
