import logging
import os
import pickle

from abc import ABC, abstractmethod
from typing import Optional

import mlflow
import numpy as np

from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models.schema_pruning import ColumnType, SchemaPruningModel

__all__ = ['Evaluator', 'SchemaPruningEvaluator']

logger = logging.getLogger()

class Evaluator(ABC):
  @abstractmethod
  def evaluate(
    self,
    data_loader: DataLoader,
    model: Module,
    step: Optional[int] = None,
  ) -> float:
    ...

class SchemaPruningEvaluator(Evaluator):
  def __init__(self, data_dir: str) -> None:
    self.all_dates = pickle.load(open(os.path.join(data_dir, 'duration_list.pkl'), 'rb'))
    self.all_timespans = pickle.load(open(os.path.join(data_dir, 'all_timespans.pkl'), 'rb'))
    self.all_scores = pickle.load(open(os.path.join(data_dir, 'all_scores.pkl'), 'rb'))

  def evaluate(
    self,
    data_loader: DataLoader,
    model: SchemaPruningModel,
    step: Optional[int] = None,
  ) -> float:
    model.eval()
    if step is not None:
      for col_type in list(ColumnType):
        mlflow.log_metric(
          f'{col_type.name} Decision Threshold', model.decision_thresholds[col_type],
          step,
        )
    all_preds = []
    total = 0
    total_pos = 0
    true_pos = 0
    pred_pos = 0
    total_ele = 0
    pos_ele = 0
    total_score = 0
    total_date = 0
    total_timespan = 0
    pos_score = 0
    pos_date = 0
    pos_timespan = 0
    pos_pred_induced = 0
    correct_gold_induced = 0
    correct_pred_induced = 0
    for batch in data_loader:
      prediction, _ = model(batch, isTrain=False)
      for d in prediction:
        data_id = d['result']['id']
        gold = d['result']['col_gold']
        pred_col = d['result']['col_pred']
        columns = d['result']['columns']
        has_lb = True
        for ii, gold_label in enumerate(gold):
          if gold_label == 1:
            total_pos += 1
            if pred_col[ii] == 1:
              true_pos += 1
            else:
              has_lb = False
          if '**induced' in columns[ii][-1]:
            if pred_col[ii] == 1:
              pos_pred_induced += 1
            if gold[ii] == 1:
              correct_gold_induced += 1
            if pred_col[ii] == 1 and gold[ii] == 1:
              correct_pred_induced += 1
        for p_col in pred_col:
          if p_col == 1:
            pred_pos += 1
          total += 1
        total_ele += 1
        if has_lb:
          pos_ele += 1
        if data_id in self.all_dates:
          total_date += 1
          if has_lb:
            pos_date += 1
        if data_id in self.all_scores:
          total_score += 1
          if has_lb:
            pos_score += 1
        if data_id in self.all_timespans:
          total_timespan += 1
          if has_lb:
            pos_timespan += 1
      all_preds.extend(prediction)
    precision = true_pos / pred_pos
    recall = true_pos / total_pos
    instance_recall = pos_ele / total_ele
    date_recall = pos_date / total_date
    score_recall = pos_score / total_score
    timespan_recall = pos_timespan / total_timespan
    induced_prec = correct_pred_induced /  (pos_pred_induced + 1e-12)
    induced_recall = correct_pred_induced /  (correct_gold_induced + 1e-12)
    logger.info(f'Precision: {true_pos}/{pred_pos} = {precision * 100}%')
    logger.info(f'Recall: {true_pos}/{total_pos} = {recall * 100}%')
    logger.info(f'Instance Recall: {pos_ele}/{total_ele} = {instance_recall * 100}%')
    logger.info(f'Precision Induced: {correct_pred_induced}/{pos_pred_induced} = {induced_prec * 100}%')
    logger.info(f'Recall Induced: {correct_pred_induced}/{correct_gold_induced} = {induced_recall * 100}%')
    logger.info(f'Date Recall: {pos_date}/{total_date} = {date_recall * 100}%')
    logger.info(f'Score Recall: {pos_score}/{total_score} = {score_recall * 100}%')
    logger.info(f'Timespan Recall: {pos_timespan}/{total_timespan} = {timespan_recall * 100}%')
    logger.info(f'Number of Positive Predictions: {pred_pos}')
    logger.info(f'Number of Total: {total}')
    if step is not None:
      mlflow.log_metric('Precision', precision, step)
      mlflow.log_metric('Recall', recall, step)
      mlflow.log_metric('Instance Recall', instance_recall, step)
      mlflow.log_metric('Date Recall', date_recall, step)
      mlflow.log_metric('Score Recall', score_recall, step)
      mlflow.log_metric('Timespan Recall', timespan_recall, step)
      mlflow.log_metric('Number of Positive Predictions', pred_pos, step)
      mlflow.log_metric('Number of Total', total, step)
    return pos_ele / total_ele, all_preds

