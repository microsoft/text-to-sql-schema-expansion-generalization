import argparse
import torch

from allennlp.models.archival import Archive, load_archive, archive_model
from allennlp.data.vocabulary import Vocabulary
from smbop.modules.relation_transformer import *
import json
from allennlp.common import Params
from smbop.models.smbop import SmbopParser
from smbop.modules.lxmert import LxmertCrossAttentionLayer
from smbop.dataset_readers.spider import SmbopSpiderDatasetReader
import itertools
import smbop.utils.node_util as node_util
import numpy as np
import numpy as np
import json
import tqdm
from allennlp.models import Model
from allennlp.common.params import *
from allennlp.data import DatasetReader, Instance
import tqdm
from allennlp.predictors import Predictor
import json
import pickle 

date_list = pickle.load(open('duration_list.pkl','rb'))
time_list = pickle.load(open('all_timespans.pkl','rb'))
score_list = pickle.load(open('all_scores.pkl','rb'))
score_unary_list = pickle.load(open('unary_score_ids.pkl', 'rb'))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_path",type=str)
    parser.add_argument("--dev_path", type=str, default="dataset/dev_data_squall_exp4.json")
    parser.add_argument("--table_path", type=str, default="dataset/table_squall_exp4.json")
    parser.add_argument("--dataset_path", type=str, default="dataset/database")
    parser.add_argument(
        "--output", type=str, default="predictions_with_vals_fixed4.txt"
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    overrides = {
        "dataset_reader": {
            "tables_file": args.table_path,
            "dataset_path": args.dataset_path,
        }
    }
    overrides["validation_dataset_reader"] = {
        "tables_file": args.table_path,
        "dataset_path": args.dataset_path,
    }
    predictor = Predictor.from_path(
        args.archive_path, cuda_device=args.gpu, overrides=overrides
    )
    print("after pred")
    total = 0 
    correct_score_unary = 0 
    correct_date = 0 
    correct_score = 0 
    total_score_unary = 0 
    total_date = 0 
    total_score = 0 

    with open(args.output, "w") as g:
        with open(args.dev_path) as f:
            dev_json = json.load(f)
            els = list()
            instances = list()
            for i, el in enumerate(tqdm.tqdm(dev_json)):
                #if i == 0:
                #    instance = predictor._dataset_reader.text_to_instance(
                #    utterance=el["question"], db_id=el["db_id"], sql= ' '.join(el["query_toks_no_value"]), sql_with_values = el["query"],raw_col=el['raw_columns']
                #    )
                #    instance_0 = instance
                #tbl = el['db_id']
                #if tbl not in date_list and tbl not in time_list and tbl not in score_list and tbl not in score_unary_list:
                #    continue
                tbl = el['db_id']
                #if tbl not in date_list and tbl not in time_list and tbl not in score_list and tbl not in score_unary_list and i > 0:
                #    continue

                instance = predictor._dataset_reader.text_to_instance(
                    utterance=el["question"], db_id=el["db_id"], sql= ' '.join(el["query_toks_no_value"]), sql_with_values = el["query"],raw_col=el['raw_columns']
                )
                instances.append(instance)


                # There is a bug that if we run with batch_size=1, the predictions are different.
                #if i == 0:
                #    instance_0 = instance
                #    continue
                

                if instance is not None:
                    predictor._dataset_reader.apply_token_indexers(instance)
                    #predictor._dataset_reader.apply_token_indexers(instance_0)
                    with torch.cuda.amp.autocast(enabled=True):
                        out = predictor._model.forward_on_instances(
                            [instance]
                        )
                        pred = out[0]["sql_list"]
                        if out[0]['reranker_acc'] == 1:
                            total += 1
                            #print(out[0]['sql_list'])
                            #print(el['query_toks'])
                        #else:
                        #    print(out[0]['sql_list'])
                        #    print(el['query_toks'])
                        #    print(out[0]['leaf_acc'])
                        #    print()
                        #    input()
                        if tbl in score_unary_list:
                            total_score_unary += 1
                            if out[0]['reranker_acc'] == 1:
                                correct_score_unary += 1
                        if tbl in score_list:
                            total_score += 1
                            if out[0]['reranker_acc'] == 1:
                                correct_score += 1
                            
                        if tbl in date_list or tbl in time_list:
                            total_date += 1
                            if out[0]['reranker_acc'] == 1:
                                correct_date += 1
                            else:
                                print(out[0]['sql_list'])
                                print(el['query_toks'])
                                print('=========')
                            #    input()

                        

                else:
                    pred = "NO PREDICTION"
                g.write(f"{pred}\t{el['db_id']}\n")
    print(correct_date, total_date)
    print(correct_score_unary, total_score_unary)
    print(correct_score, total_score)
    print(total)

if __name__ == "__main__":
    main()
