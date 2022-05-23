import logging

from enum import Enum
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .helpers import Encoder_rnn, MatchAttn

logger = logging.getLogger()

class ColumnType(Enum):
  ANY = 1
  INDUCED = 2
  OTHER = 3

class SchemaPruningModel(nn.Module):
  def __init__(
    self,
    args,
    vocab,
    use_cuda: bool,
    dataset: List[Dict[str, Any]] = None,
    tgt_avg_num_columns_per_example_offsets: Dict[ColumnType, float] = dict(),
    use_separate_decision_thresholds: bool = False,
    **kwargs,
  ) -> None:
    super(SchemaPruningModel, self).__init__()
    self.args = args
    self.vocab = vocab
    self.device = torch.device('cuda' if use_cuda else 'cpu')

    self._transformer_dim = 256

    self._mlp_dim = 128
    self._mlp_dropout = 0.2

    self._bert_model = AutoModel.from_pretrained('bert-base-uncased')
    self._q_bert_model = AutoModel.from_pretrained('bert-base-uncased')
    self._q_bert_model.embeddings = self._bert_model.embeddings

    self._bert_w_project = nn.Linear(self._bert_model.pooler.dense.in_features, self._transformer_dim)
    self._bert_c_project = nn.Linear(self._bert_model.pooler.dense.in_features, self._transformer_dim)

    self._type_final = torch.nn.Sequential(
      nn.Linear(self._transformer_dim, self._mlp_dim),
      nn.Dropout(self._mlp_dropout),
      nn.ReLU(),
      nn.Linear(self._mlp_dim, 1),
    )

    self._bilstm_dim = self._transformer_dim // 2
    self.c_attn_w = MatchAttn(self._bilstm_dim * 2, identity=True)
    self.w_attn_c = MatchAttn(self._bilstm_dim * 2, identity=True)

    self._q_bilstm_2 = Encoder_rnn(self.args, self._bilstm_dim * 4, self._bilstm_dim)
    self._c_bilstm_2 = Encoder_rnn(self.args, self._bilstm_dim * 4, self._bilstm_dim)

    self._loss_fn = nn.BCELoss(reduction='none')

    self.epoch = 0
    self.decision_thresholds = dict()
    if dataset is not None:
      avg_num_columns_per_example = _compute_avg_num_columns_per_example(dataset)
      self.tgt_avg_num_columns_per_example = {}
      for col_type in list(ColumnType):
        self.tgt_avg_num_columns_per_example[col_type] = avg_num_columns_per_example[col_type] \
          + tgt_avg_num_columns_per_example_offsets.get(col_type, 0)
    else:
      self.tgt_avg_num_columns_per_example = None
    self.use_separate_decision_thresholds = use_separate_decision_thresholds
    self.to(self.device)

  @staticmethod
  def load(model_file: str, use_cuda: bool):
    if use_cuda:
      device = torch.device('cuda')
      devices_map = {}
    else:
      device = torch.device('cpu')
      devices_map = {'cuda:0': 'cpu'}
    model = torch.load(model_file, map_location=devices_map)
    model.device = device
    model.to(device)
    return model

  def _bert_features(self, batch):
    instances = batch['instances']
    word_seq_max_len = batch['word_seq_max_len'].to(self.device)
    col_seq_max_len = batch['col_seq_max_len'].to(self.device)
    all_input_ids = batch['all_input_ids'].to(self.device)
    all_input_type_ids = batch['all_input_type_ids'].to(self.device)
    all_input_mask = batch['all_input_mask'].to(self.device)
    all_word_end_mask = batch['all_word_end_mask'].to(self.device)
    all_col_end_mask = batch['all_col_end_mask'].to(self.device)
    
    bert_output = self._bert_model(all_input_ids, token_type_ids=all_input_type_ids, attention_mask=all_input_mask)
    features = bert_output['last_hidden_state']
    
    bert_word_features = features.masked_select(all_word_end_mask.to(torch.bool).unsqueeze(-1)).reshape(len(instances), word_seq_max_len, features.shape[-1])
    bert_col_features = features.masked_select(all_col_end_mask.to(torch.bool).unsqueeze(-1)).reshape(len(instances), col_seq_max_len, features.shape[-1])

    # BERT encoding for question and table
    wvecs = self._bert_w_project(bert_word_features)
    cvecs = self._bert_c_project(bert_col_features)

    w_mask = batch['word_mask'].to(self.device)
    c_mask = batch['col_mask'].to(self.device)

    # Attention
    wcontext, alpha_w = self.w_attn_c(wvecs, cvecs, c_mask)
    ccontext, alpha_c = self.c_attn_w(cvecs, wvecs, w_mask)

    # Supervised Attn
    wvecs = torch.cat((wvecs, wcontext), 2)
    cvecs = torch.cat((cvecs, ccontext), 2)

    w_len = batch['word_mask'].data.eq(0).long().sum(1).numpy().tolist()
    wvecs = self._q_bilstm_2(wvecs, w_len)
    c_len = batch['col_mask'].data.eq(0).long().sum(1).numpy().tolist()
    cvecs = self._c_bilstm_2(cvecs, c_len)

    return wvecs, cvecs

  def forward(self, batch, isTrain=True):
    _, cvecs = self._bert_features(batch) # [B, C, H]
    if cvecs.size(1) == 1:
      potentials = self._type_final(cvecs).squeeze(1)
      print(potentials.size())
    else:
      potentials = self._type_final(cvecs).squeeze(1).squeeze(2) # [B, C]
    instances = batch['instances']

    if isTrain:
      labels = batch['col_lbs'].to(self.device) # [B, C]
      col_lbs_mask = 1 - batch['col_lbs_mask'].to(self.device) # [B]
      loss = self._loss_fn(torch.nn.Sigmoid()(potentials), labels.float()) # [B, C]
      loss = torch.sum(loss * col_lbs_mask, dim=1) / torch.sum(col_lbs_mask, dim=1) # [B]
      loss = torch.mean(loss)
      return loss
    else:
      logprobs = []
      pred_data = list()
      for i in range(len(batch['sqls'])):
        potential = torch.sigmoid(potentials[i]).data.cpu().numpy()
        num_cols = len(instances[i]['col_lbs'])
        potential = potential[:num_cols]

        str_scores = list()
        scores = list()
        for ii, item in enumerate(potential):
          str_scores.append(str(item))
          scores.append(item)
        
        heu_output = self.args.match_dict[instances[i]["nt"]]
        heu_output = [int(ele[1:]) - 1 for ele in heu_output]

        pred_col = []
        indexs = sorted(range(len(scores)), reverse=True, key=lambda k: scores[k])
        indexs = [idx for idx in indexs if idx not in heu_output]

        for ii in range(len(scores)):
          if self.use_separate_decision_thresholds:
            col_type = ColumnType.ANY
          #elif '**induced' in instances[i]['columns'][ii][-1]:
          #  col_type = ColumnType.INDUCED
          #else:
          col_type = ColumnType.OTHER
          if self.decision_thresholds is None:
            threshold = None
          else:
            threshold = self.decision_thresholds.get(col_type, None)
            if '**induced' in instances[i]['columns'][ii][-1]:
            # or '**expanded' in instances[i]['columns'][ii][-1]:
              threshold = 0.3
            #elif '**expanded' in instances[i]['columns'][ii][-1]:
            #  threshold = 0.1
          if ii in heu_output:
            pred_col.append(1)
          elif threshold is not None and scores[ii] >= threshold:
            pred_col.append(1)
          elif threshold is None and ii in indexs[:3]:
            pred_col.append(1)
          else:
            pred_col.append(0)
        
        pred_data.append({
          'table_id': instances[i]["tbl"],
          'result': {
            'col_pred': pred_col,
            'col_gold': instances[i]["col_lbs"],
            'score': ' '.join(str_scores),
            'id': instances[i]["nt"],
            'columns': instances[i]['columns'],
            'nl': ' '.join(instances[i]['nl'])
          }
        })

      return pred_data, logprobs

  def set_decision_thresholds(self, data_loader: DataLoader, num_batches: Optional[int] = None):
    self.eval()
    self.decision_thresholds = _compute_decision_thresholds(self, data_loader, num_batches)

def _compute_avg_num_columns_per_example(dataset: List[Dict[str, Any]]) -> Dict[ColumnType, float]:
  logger.info('Counting the average number of columns per example.')
  total_num_columns = 0
  total_num_induced_columns = 0
  total_num_other_columns = 0
  total_num_examples = 0
  for instance in tqdm(dataset):
    column_idxs = [i for i, l in enumerate(instance['col_lbs']) if l == 1]
    total_num_columns += len(column_idxs)
    total_num_examples += 1
    for c in column_idxs:
      #if '**induced' in instance['columns'][c][-1] or '**expanded' in instance['columns'][c][-1]:
      if '**induced' in instance['columns'][c][-1]:
        total_num_induced_columns += 1
      else:
        total_num_other_columns += 1
  avg_num_columns_per_example = total_num_columns / total_num_examples
  avg_num_induced_columns_per_example = total_num_induced_columns / total_num_examples
  avg_num_other_columns_per_example = total_num_other_columns / total_num_examples
  logger.info(f'Got {avg_num_columns_per_example} average number of columns per example.')
  logger.info(f'Got {avg_num_induced_columns_per_example} average number of induced columns per example.')
  logger.info(f'Got {avg_num_other_columns_per_example} average number of other columns per example.')
  return {
    ColumnType.ANY: avg_num_columns_per_example,
    ColumnType.INDUCED: avg_num_induced_columns_per_example,
    ColumnType.OTHER: avg_num_other_columns_per_example,
  }

def _compute_decision_thresholds(
  model: SchemaPruningModel,
  data_loader: DataLoader,
  num_batches: Optional[int] = None,
) -> Dict[ColumnType, float]:
  logger.info('Computing the decision threshold.')

  logger.info('Collecting predicted scores.')
  predicted_scores = []
  for b, batch in enumerate(tqdm(data_loader)):
    if num_batches is not None and b > num_batches:
      break
    prediction, _ = model(batch, isTrain=False)
    for p in prediction:
      scores = [float(sc) for sc in p['result']['score'].split(' ')]
      columns = p['result']['columns']
      predicted_scores.append((scores, columns))
  
  def _compute_avg_num_columns_per_example(col_type: ColumnType, threshold: float) -> float:
    num_positives = 0
    for scores, columns in predicted_scores:
      if col_type == ColumnType.ANY:
        num_positives += len([s for s in scores if s >= threshold])
      else:
        for c, score in enumerate(scores):
          if col_type == ColumnType.INDUCED:
            keep = '**induced' in columns[c][-1]
          else:
            keep = '**induced' not in columns[c][-1]
            #keep = '**induced' not in columns[c][-1] and '**expanded' not in columns[c][-1]
          if keep and score >= threshold:
            num_positives += 1
    return num_positives / len(predicted_scores)

  def _binary_search_for_threshold(
    col_type: ColumnType,
    search_step: int,
    threshold_lb: float,
    threshold_ub: float,
  ) -> float:
    threshold_midpoint = (threshold_ub + threshold_lb) / 2
    max_search_steps = 100
    if search_step > max_search_steps:
      return threshold_midpoint
    avg_num_columns_per_example = _compute_avg_num_columns_per_example(col_type, threshold_midpoint)
    if abs(avg_num_columns_per_example - model.tgt_avg_num_columns_per_example[col_type]) < 0.01:
      return threshold_midpoint
    if avg_num_columns_per_example < model.tgt_avg_num_columns_per_example[col_type]:
      return _binary_search_for_threshold(col_type, search_step + 1, threshold_lb, threshold_midpoint)
    return _binary_search_for_threshold(col_type, search_step + 1, threshold_midpoint, threshold_ub)

  thresholds = dict()
  for col_type in list(ColumnType):
    thresholds[col_type] = _binary_search_for_threshold(
      col_type,
      search_step=0,
      threshold_lb=0.0,
      threshold_ub=1.0,
    )
  logger.info(f'Using thresholds: {thresholds}.')
  return thresholds
