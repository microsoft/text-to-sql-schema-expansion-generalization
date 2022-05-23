import argparse
import json
import logging
import os
import pickle
import time

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import torch
import transformers
import mlflow
import numpy as np

from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from .dataset import create_data_loader, load_dataset
from .evaluation import *
from .models.schema_pruning import ColumnType, SchemaPruningModel
from .vocab import build_vocab

logger = logging.getLogger()
torch.manual_seed(0)
np.random.seed(0)

_Module = TypeVar('_Module', bound=Module)

def _model_parameters(model: Module) -> Tuple[List[Parameter], List[Parameter]]:
  params = []
  params_bert = []
  for name, param in model.named_parameters():
    if param.requires_grad:
      if 'bert_model' in name:
        params_bert.append(param)
      else:
        params.append(param)
  return params, params_bert

class Experiment(ABC):
  @abstractmethod
  def name(self) -> str:
    ...
  
  @abstractmethod
  def model_class(self) -> Type[_Module]:
    ...
  
  @abstractmethod
  def num_training_steps(self) -> int:
    ...

  @abstractmethod
  def optimizer(self, model: Module) -> Optimizer:
    ...

  @abstractmethod
  def lr_scheduler(self, num_training_steps: int) -> Optional[_LRScheduler]:
    ...
  
  @abstractmethod
  def evaluator(self, data_dir: str, gold_decode: bool = False) -> Evaluator:
    ...

class SchemaPruningExperiment(Experiment):
  def name(self) -> str:
    return 'schema-pruning'
  
  def model_class(self) -> Type[SchemaPruningModel]:
    return SchemaPruningModel
  
  def num_training_steps(self) -> int:
    return 5000

  def optimizer(self, model: Module) -> Optimizer:
    params, params_bert = _model_parameters(model)
    return torch.optim.AdamW(
      [
        {'params': params, 'lr': 1e-4},
        {'params': params_bert, 'lr': 1e-5}
      ],
      lr=1e-4,
      weight_decay=0.01,
    )
  
  def lr_scheduler(self, num_training_steps: int) -> Optional[_LRScheduler]:
    return transformers.optimization.get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=100,
      num_training_steps=num_training_steps,
    )

  def evaluator(self, data_dir: str, _: bool = False) -> SchemaPruningEvaluator:
    return SchemaPruningEvaluator(data_dir)


def _get_or_create_dir(home_dir: str, path: str) -> str:
  path = os.path.join(home_dir, path)
  if not os.path.exists(path):
    os.makedirs(path)
  return path

def _initialize_logging(args: Dict[str, Any]) -> None:
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')

  # Configure console logging.
  console_log_handler = logging.StreamHandler()
  console_log_handler.setFormatter(formatter)
  logger.addHandler(console_log_handler)

  # Configure file logging.
  log_dir = _get_or_create_dir(args.home_dir, 'log')
  log_file = os.path.join(log_dir, args.log_file)
  file_log_handler = logging.FileHandler(log_file, 'a')
  file_log_handler.setFormatter(formatter)
  logger.addHandler(file_log_handler)

  # Initialize MLFlow for logging metrics.
  mlflow.start_run()

  # Log some information about the current experiment parameters.
  mlflow.log_param('cuda', args.cuda)

def _deinitialize_logging() -> None:
  mlflow.end_run()

def _load_datasets(
  train_file: str,
  dev_file: Optional[str],
  test_file: Optional[str],
  num_train_samples: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
  # Load the train dataset.
  train_dataset = load_dataset(train_file)
  if num_train_samples != 0:
    train_dataset = train_dataset[:num_train_samples]

  # Load the dev dataset.
  if dev_file is not None:
    dev_dataset = load_dataset(dev_file)
  else:
    dev_dataset = train_dataset[-500:]
    train_dataset = train_dataset[:-500]

  # Load the test dataset.
  if test_file is not None:
    test_dataset = load_dataset(test_file)
  else:
    test_dataset = None

  return train_dataset, dev_dataset, test_dataset

def _create_data_loaders(
  train_dataset: List[Dict[str, Any]],
  dev_dataset: List[Dict[str, Any]],
  test_dataset: Optional[List[Dict[str, Any]]],
  vocabulary: Dict[str, Dict[str, int]],
  tokenizer: PreTrainedTokenizer,
  batch_size: int,
  num_workers: int,
  cuda: bool,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
  train_loader = create_data_loader(
    train_dataset,
    vocabulary,
    tokenizer,
    batch_size,
    num_workers=num_workers,
    is_train=True,
    pin_memory=cuda,
  )
  dev_loader = create_data_loader(
    dev_dataset,
    vocabulary,
    tokenizer,
    batch_size=batch_size,
    num_workers=num_workers,
    is_train=False,
    pin_memory=cuda,
  )
  if test_dataset is not None:
    test_loader = create_data_loader(
      test_dataset,
      vocabulary,
      tokenizer,
      batch_size=batch_size,
      num_workers=num_workers,
      is_train=False,
      pin_memory=cuda,
    )
  else:
    test_loader = None
  return train_loader, dev_loader, test_loader


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='SM Internship Tabular Data')
  parser.add_argument('--experiment', type=str, default='schema-pruning')
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--data-workers', type=int, default=8)
  parser.add_argument('--home-dir', type=str, default=os.path.join(os.getenv('AMLT_DATA_DIR', os.getcwd()), os.pardir))
  parser.add_argument('--train-file', type=str, default='squall/train_data_prune.json')
  parser.add_argument('--dev-file', type=str, default='squall/dev_data_prune.json')
  parser.add_argument('--test-file', type=str, default='squall/dev_data_prune.json')
  parser.add_argument('--pred-file', type=str, default='log/schema_filter_dev.json')

  parser.add_argument('--batch-size', type=int, default=16)
  parser.add_argument('--num-epochs', type=int, default=20)
  parser.add_argument('--input-size', type=int, default=300)
  parser.add_argument('--hidden-size', type=int, default=300)
  parser.add_argument('--num-layers', type=int, default=1)
  parser.add_argument('--dropout', type=int, default=0.2)

  parser.add_argument('--tgt-avg-num-induced-columns-per-example-offset', type=float, default=4)
  parser.add_argument('--tgt-avg-num-other-columns-per-example-offset', type=float, default=4)
  parser.add_argument('--use-separate-decision-thresholds', action='store_true', default=False)

  parser.add_argument('--resume', action='store_true', default=False)
  parser.add_argument('--test', action='store_true', default=False)
  parser.add_argument('--save-model', type=str, default='sm-release/log/schema_pruning.pt')
  parser.add_argument('--load-model', type=str, default='sm-release/log/schema_pruning.pt')
  parser.add_argument('--log-file', type=str, default = 'log_file_rank.log')

  parser.add_argument('--dec-loss', action='store_true', default=False)
  parser.add_argument('--enc-loss', action='store_true', default=False)
  parser.add_argument('--aux-col', action='store_true', default=False)
  parser.add_argument('--gold-decode', action='store_true', default=False)
  parser.add_argument('--gold-attn', action='store_true', default=False)
  parser.add_argument('--sample', type=int, default=50000)
  parser.add_argument('--bert', action='store_true', default=False)

  args = parser.parse_args()
  if args.experiment == 'schema-pruning':
    experiment = SchemaPruningExperiment()
  else:
    raise ValueError(f'Invalid experiment "{args.experiment}" provided.')
  logger.info(f'Running "{experiment.name()}" experiment.')

  # Preprocess the arguments.
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  log_dir = _get_or_create_dir(args.home_dir, 'sm-release/log')
  data_dir = _get_or_create_dir(args.home_dir, 'sm-release/data')
  evaluator = experiment.evaluator(data_dir, args.gold_decode)

  _initialize_logging(args)

  # Load all datasets.
  train_dataset, dev_dataset, test_dataset = _load_datasets(
    os.path.join(data_dir, args.train_file),
    os.path.join(data_dir, args.dev_file) if args.dev_file is not None else args.dev_file,
    os.path.join(data_dir, args.test_file) if args.test_file is not None else args.test_file,
    num_train_samples=args.sample,
  )
  vocabulary = build_vocab(train_dataset, cutoff=5) 
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  train_loader, dev_loader, test_loader = _create_data_loaders(
    train_dataset,
    dev_dataset,
    test_dataset,
    vocabulary,
    tokenizer,
    batch_size=args.batch_size,
    num_workers=args.data_workers,
    cuda=args.cuda,
  )

  model_class = experiment.model_class()
  if args.test:
    model_file = os.path.join(args.home_dir, args.load_model)
    
    device = torch.device('cuda')
    devices_map = {}
    bn_state_dict = torch.load(model_file)
    match_dict_path = os.path.join(args.home_dir, 'sm-release/data/squall/match_dict.pkl')
    setattr(args, 'match_dict', pickle.load(open(match_dict_path, 'rb')))
    model = model_class(
      args,
      vocabulary,
      args.cuda,
      data_dir=data_dir,
      dataset=train_dataset,
      tgt_avg_num_columns_per_example_offsets={
        ColumnType.ANY: args.tgt_avg_num_other_columns_per_example_offset,
        ColumnType.INDUCED: args.tgt_avg_num_induced_columns_per_example_offset,
        ColumnType.OTHER: args.tgt_avg_num_other_columns_per_example_offset,
    })
    model.load_state_dict(bn_state_dict)
    model.device = device
    model.to(device)
    if hasattr(model, 'set_decision_thresholds'):
      model.set_decision_thresholds(dev_loader)
    
    f1, pred = evaluator.evaluate(test_loader, model)
    with open(os.path.join(args.home_dir, args.pred_file), 'w') as f:
      json.dump(pred, f, indent=2)
    exit()
  elif args.resume:
    model_file = os.path.join(args.home_dir, args.load_model)
    model = model_class(
      args,
      vocabulary,
      args.cuda,
      data_dir=data_dir,
      dataset=train_dataset,
      tgt_avg_num_columns_per_example_offsets={
        ColumnType.ANY: args.tgt_avg_num_other_columns_per_example_offset,
        ColumnType.INDUCED: args.tgt_avg_num_induced_columns_per_example_offset,
        ColumnType.OTHER: args.tgt_avg_num_other_columns_per_example_offset,
    })
    bn_state_dict = torch.load(model_file)
    model.load_state_dict(bn_state_dict)
    device = torch.device('cuda')
    model.device = device
    model.to(device)
    #model = model_class.load(model_file, args.cuda)
  else:
    match_dict_path = os.path.join(args.home_dir, 'sm-release/data/squall/match_dict.pkl')
    setattr(args, 'match_dict', pickle.load(open(match_dict_path, 'rb')))
    model = model_class(
      args,
      vocabulary,
      args.cuda,
      data_dir=data_dir,
      dataset=train_dataset,
      tgt_avg_num_columns_per_example_offsets={
        ColumnType.ANY: args.tgt_avg_num_other_columns_per_example_offset,
        ColumnType.INDUCED: args.tgt_avg_num_induced_columns_per_example_offset,
        ColumnType.OTHER: args.tgt_avg_num_other_columns_per_example_offset,
      })
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(args.home_dir, args.save_model))
    

  

  checkpoint_steps = 500
  num_training_steps = experiment.num_training_steps()
  optimizer = experiment.optimizer(model)
  lr_scheduler = experiment.lr_scheduler(num_training_steps)

  logger.info('start training:')
  model.train()
  epoch = 0
  step = 0
  print_loss_total = 0
  epoch_loss_total = 0
  start = time.time()
  max_f1 = -np.inf

  while True:
    if step > num_training_steps:
      break
    print_loss_total = 0

    logger.info('start epoch:%d' % epoch)
    model.epoch = epoch

    for idx, batch in enumerate(tqdm(train_loader)):
      loss = model(batch)
      loss_value = float(loss.data.cpu().numpy())
      mlflow.log_metric('Loss', loss_value, step)
      #print(loss_value)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if lr_scheduler is not None:
        lr_scheduler.step()

      clip_grad_norm_(model.parameters(), 1)
      print_loss_total += loss_value
      epoch_loss_total += loss_value
      if idx % checkpoint_steps == 0 and idx > 0:
        if hasattr(model, 'set_decision_thresholds'):
          model.set_decision_thresholds(dev_loader, num_batches=20)
        f1, _ = evaluator.evaluate(dev_loader, model, step)
        #f1 = 0 
        model.train()
        print_loss_avg = print_loss_total / checkpoint_steps
        logger.info('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time()- start))
        print_loss_total = 0
        if f1 > max_f1:
          max_f1 = f1
        
          state_dict = model.state_dict()
        print(os.path.join(args.home_dir, args.save_model + '_' + str(epoch) + '_' + str(idx) + '.pt'))
        torch.save(model.state_dict(), os.path.join(args.home_dir, args.save_model + '_' + str(epoch) + '_' + str(idx) + '.pt'))

      step += 1
    epoch += 1
  if hasattr(model, 'set_decision_thresholds'):
    model.set_decision_thresholds(dev_loader)
  torch.save(model.state_dict(), os.path.join(args.home_dir, args.save_model + '_' + str(epoch) + '_' + str(idx) + '.pt'))
  _deinitialize_logging()
