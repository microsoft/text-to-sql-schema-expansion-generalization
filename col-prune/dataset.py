import json

from typing import Any, Dict, List

import numpy as np

from torch import from_numpy
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer

from .utils import parse_number
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def load_dataset(filename) -> List[Dict[str, Any]]:
  # Load the data instances from the JSON file.
  with open(filename) as f:
    dataset = json.load(f)
  for instance in dataset:
    has_number = False
    numbers = []
    for x in instance['nl']:
      numbers.append(parse_number(x))
      if numbers[-1] is not None:
        has_number = True
    instance['numbers'] = numbers
    instance['has_number'] = has_number
  return dataset

class WikiTableDataset(Dataset):
  def __init__(self, examples, vocab):
    self.examples = examples
    self.vocab = vocab

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, index):
    instance = self.examples[index]
    return {
      'sql': instance.get('first_stage', []) + [['Keyword', ['<EOS>'], []]],
      'instance': instance,
    }

def create_data_loader(
  dataset: List[Dict[str, Any]],
  vocabulary: Dict[str, Dict[str, int]],
  tokenizer: PreTrainedTokenizer,
  batch_size: int,
  num_workers: int,
  is_train: bool,
  pin_memory: bool,
) -> DataLoader:
  dataset = WikiTableDataset(dataset, vocabulary)
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=is_train,
    num_workers=num_workers,
    collate_fn=_batch,
    pin_memory=pin_memory,
    persistent_workers=True,
  )

def _batch(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  sqls = [e['sql'] for e in examples]
  instances = [e['instance'] for e in examples]

  pruning_dataset = 'col_lbs' in instances[0]
  col_lbs_max_len = max([len(ins['col_lbs']) for ins in instances]) if pruning_dataset else 0
  col_lbs = []
  col_lbs_mask = []

  word_seq_max_len = max([len(ins['nl']) for ins in instances])
  col_seq_max_len = max([len(ins['columns']) for ins in instances])

  all_input_ids = []
  all_input_type_ids = []
  all_input_mask = []
  all_word_end_mask = []
  all_col_end_mask = []

  word_masks = []
  col_masks = []

  subword_max_len = 0
  for ins in instances:
    if pruning_dataset:
      col_lbs.append(ins['col_lbs'] + [0] * (col_lbs_max_len - len(ins['col_lbs'])))
      col_lbs_mask.append([0 for _ in ins['col_lbs']] + [1] * (col_lbs_max_len - len(ins['col_lbs'])))

    question_words = ins['nl']
    tokens = []
    token_types = []
    word_mask = []
    col_mask = []
    word_end_mask = []
    col_end_mask = []
    tokens.append('[CLS]')
    token_types.append(0)
    word_end_mask.append(0)
    col_end_mask.append(0)
    
    for word in question_words:
      word_mask.append(0)
      word_tokens = tokenizer.tokenize(word)
      if len(word_tokens) == 0:
        word_tokens = ['.']
      for _ in range(len(word_tokens)):
        word_end_mask.append(0)
        col_end_mask.append(0)
        token_types.append(0)
      word_end_mask[-1] = 1
      tokens.extend(word_tokens)
    tokens.append('[SEP]')
    word_end_mask.append(0)
    col_end_mask.append(0)
    token_types.append(0)

    for col in ins['columns']:
      col_mask.append(0)
      col_tokens = tokenizer.tokenize(col[0])
      if len(col_tokens) == 0:
        col_tokens = ['.']
      for _ in range(len(col_tokens)):
        word_end_mask.append(0)
        col_end_mask.append(0)
        token_types.append(1)
      col_end_mask[-1] = 1
      tokens.extend(col_tokens)
      tokens.append('[SEP]')
      word_end_mask.append(0)
      col_end_mask.append(0)
      token_types.append(1)
    
    word_masks.append(word_mask)
    col_masks.append(col_mask)

    # pad to sequence length for every sentence
    word_end_mask.extend([1] * (word_seq_max_len - len(ins['nl'])))
    col_end_mask.extend([1] * (col_seq_max_len - len(ins['columns'])))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_type_ids = token_types
    input_mask = [1] * len(input_ids)

    subword_max_len = max(max(subword_max_len, len(word_end_mask) + 1), len(col_end_mask) + 1)

    all_input_ids.append(input_ids)
    all_input_type_ids.append(input_type_ids)
    all_input_mask.append(input_mask)
    all_word_end_mask.append(word_end_mask)
    all_col_end_mask.append(col_end_mask)

  for i in range(len(all_input_ids)):
    all_input_ids[i].extend([0] * (subword_max_len - len(all_input_ids[i])))
    all_input_type_ids[i].extend([0] * (subword_max_len - len(all_input_type_ids[i])))
    all_input_mask[i].extend([0] * (subword_max_len - len(all_input_mask[i])))
    all_word_end_mask[i].extend([0] * (subword_max_len - len(all_word_end_mask[i])))
    all_col_end_mask[i].extend([0] * (subword_max_len - len(all_col_end_mask[i])))

  for mask in word_masks:
    mask.extend([1] * (word_seq_max_len - len(mask)))
  for mask in col_masks:
    mask.extend([1] * (col_seq_max_len - len(mask)))
  
  word_masks = np.array([np.array(m) for m in word_masks])
  col_masks = np.array([np.array(m) for m in col_masks])

  return {
    'sqls': sqls,
    'instances': instances,
    'col_lbs': from_numpy(np.ascontiguousarray(col_lbs)),
    'col_lbs_mask': from_numpy(np.ascontiguousarray(col_lbs_mask)),
    'word_mask': from_numpy(np.ascontiguousarray(word_masks)),
    'col_mask': from_numpy(np.ascontiguousarray(col_masks)),
    'word_seq_max_len': from_numpy(np.ascontiguousarray([word_seq_max_len])),
    'col_seq_max_len': from_numpy(np.ascontiguousarray([col_seq_max_len])),
    'all_input_ids': from_numpy(np.ascontiguousarray(all_input_ids)),
    'all_input_type_ids': from_numpy(np.ascontiguousarray(all_input_type_ids)),
    'all_input_mask': from_numpy(np.ascontiguousarray(all_input_mask)),
    'all_word_end_mask': from_numpy(np.ascontiguousarray(all_word_end_mask)),
    'all_col_end_mask': from_numpy(np.ascontiguousarray(all_col_end_mask)),
  }
