import argparse
import json
import logging
import os
import time

from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch

from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from dataset import create_data_loader, load_dataset
from model import TableParser
from vocab import build_vocab

logger = logging.getLogger()

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
    batch_size=16,
    num_workers=num_workers,
    is_train=False,
    pin_memory=cuda,
  )
  if test_dataset is not None:
    test_loader = create_data_loader(
      test_dataset,
      vocabulary,
      tokenizer,
      batch_size=16,
      num_workers=num_workers,
      is_train=False,
      pin_memory=cuda,
    )
  else:
    test_loader = None
  return train_loader, dev_loader, test_loader


def evaluate(data_loader, model, evaluator, gold_decode=False):
    lf_accu = 0
    all_accu = 0
    total = 0
    all_preds = list()
    log_probs = []

    for idx, batch in enumerate(tqdm(data_loader)):
        model.eval()
        
        prediction, _log_probs = model(batch, isTrain=False, gold_decode=gold_decode)
        log_probs.extend(_log_probs)
        for d in prediction:
            total += 1
            if d['result'][0]['sql'] == d['result'][0]['tgt']:
                lf_accu += 1
                prediction[0]['correct'] = 1
            else:
                prediction[0]['correct'] = 0
        all_preds.extend(prediction)

    perplexity = 2 ** (-np.average(log_probs) / np.log(2.))
    logger.info('logical form accurate: {}/{} = {}%'.format(lf_accu, total, lf_accu / total * 100))
    logger.info('perplexity: {}'.format(perplexity))
    if gold_decode:
        return -perplexity, all_preds
    else:
        return lf_accu, all_preds


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Table Parse')
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--data-workers', type=int, default=2)
  parser.add_argument('--home-dir', type=str, default=os.path.join(os.getenv('AMLT_DATA_DIR', os.getcwd()), os.pardir))
  parser.add_argument('--train-file', type=str, default='squall/train_data_parser.json')
  parser.add_argument('--dev-file', type=str, default='squall/dev_data_parser.json')
  parser.add_argument('--test-file', type=str, default='squall/dev_data_parser.json')
  parser.add_argument('--pred-file', type=str, default='pred_data_parser.json')
  parser.add_argument('--batch-size', type=int, default=16)
  parser.add_argument('--num-epochs', type=int, default=30)
  parser.add_argument('--input-size', type=int, default=300)
  parser.add_argument('--hidden-size', type=int, default=300)
  parser.add_argument('--num-layers', type=int, default=1)
  parser.add_argument('--dropout', type=int, default=0.2)
  parser.add_argument('--resume', action='store_true', default=False)
  parser.add_argument('--test', action='store_true', default=False)
  parser.add_argument('--save-model', type=str, default='parsing')
  parser.add_argument('--load-model', type=str, default='parsing')
  parser.add_argument('--log-file', type=str, default = 'log_file_parse.log')

  parser.add_argument('--dec-loss', action='store_true', default=False)
  parser.add_argument('--enc-loss', action='store_true', default=False)
  parser.add_argument('--aux-col', action='store_true', default=False)
  parser.add_argument('--gold-decode', action='store_true', default=False)
  parser.add_argument('--gold-attn', action='store_true', default=False)
  parser.add_argument('--sample', type=int, default=50000)
  parser.add_argument('--bert', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=30)

  args = parser.parse_args()
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)


  # Preprocess the arguments.
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  log_dir = _get_or_create_dir(args.home_dir, 'log')
  data_dir = _get_or_create_dir(args.home_dir, 'data')
  args.data_dir = data_dir
  _initialize_logging(args)

  # Load all datasets.
  train_dataset, dev_dataset, test_dataset = _load_datasets(
    os.path.join(data_dir, args.train_file),
    os.path.join(data_dir, args.dev_file) if args.dev_file is not None else args.dev_file,
    os.path.join(data_dir, args.test_file) if args.test_file is not None else args.test_file,
    num_train_samples=args.sample,
  )
  vocabulary = build_vocab(train_dataset, cutoff=5)
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
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

  evaluator = None
  device = torch.device('cuda' if args.cuda else 'cpu')
  if args.test:
    if not args.cuda:
      model_file = os.path.join(args.home_dir, args.load_model)
      model = torch.load(model_file, map_location={'cuda:0': 'cpu'})
      model.device = device
    else:
      model_file = os.path.join(args.home_dir, args.load_model)
      model = torch.load(model_file)
    model.to(device)
    f1, pred = evaluate(test_loader, model, evaluator, args.gold_decode)
    pred_file = os.path.join(args.home_dir, args.pred_file)
    with open(pred_file, "w") as f:
      json.dump(pred, f, indent=2)

    exit()

  elif args.resume:
    if not args.cuda:
      model = torch.load(args.load_model, map_location={'cuda:0': 'cpu'})
      model.device = device
    else:
      model = torch.load(args.load_model)
    model.to(device)

  else:
    model = TableParser(args, vocabulary, device)
    model.to(device)

  start_epoch = 0

  params = []
  params_bert = []
  for name, param in model.named_parameters():
    if param.requires_grad:
      if 'bert_model' in name:
        params_bert.append(param)
      else:
        params.append(param)
  optimizer = [torch.optim.Adamax(params, lr=1e-3)]
  if len(params_bert):
    optimizer.append(torch.optim.Adamax(params_bert, lr=1e-5))

  logger.info('start training:')
  print_loss_total = 0
  epoch_loss_total = 0
  start = time.time()
  max_f1 = -np.inf

  ### model training
  for epoch in range(start_epoch, args.num_epochs):
    print_loss_total = 0

    logger.info('start epoch:%d' % epoch)
    model.train()
    model.epoch = epoch

    for idx, batch in enumerate(tqdm(train_loader)):
      loss = model(batch)
      loss = sum(loss)/ len(batch['instances'])

      for opt in optimizer:
        opt.zero_grad()
      loss.backward()
      for opt in optimizer:
        opt.step()

      clip_grad_norm_(model.parameters(), 5)
      print_loss_total += loss.data.cpu().numpy()
      epoch_loss_total += loss.data.cpu().numpy()
      checkpoint = 50

      if idx % checkpoint == 0 and idx != 0:
        f1, pred = evaluate(dev_loader, model, evaluator, args.gold_decode)
        model.train()
        print_loss_avg = print_loss_total / checkpoint
        logger.info('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time()- start))
        print_loss_total = 0
        if f1 > max_f1:
          max_f1 = f1
          with open(os.path.join(args.home_dir, args.pred_file), "w") as f:
            json.dump(pred, f, indent=2)
          torch.save(model, os.path.join(args.home_dir, args.save_model + '_' + str(epoch) + '_' + str(idx) + '.pt'))
  _deinitialize_logging()
