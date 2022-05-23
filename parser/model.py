import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import json
import copy
import math
import numpy as np
from utils import parse_number, get_cells, best_match, attention_matrix, attention_matrix_col
from transformers import BertTokenizer, BertModel
import os

def l2_loss(x, y):
    ret = 0.
    for i in range(len(y)):
        diff = x[i] - y[i]
        ret = ret + diff * diff
    return [0.5 * ret]

def ce_loss(x, y):
    ret = 0.
    for i in range(len(y)):
        if y[i] > 0.:
            ret = ret + y[i] * torch.log(x[i] + 1e-8)
    return [-.2 * ret]

def mul_loss(x, y):
    ret = 0.
    for i in range(len(y)):
        if y[i] > 0.:
            ret = ret + x[i]
    if ret == 0:
        return []
    ret = ret + 1e-8
    return [-1. * torch.log(ret)]


class TableParser(nn.Module):


    def __init__(self, args, vocab, device):
        super(TableParser, self).__init__()
        ## Model parameters

        self._batch_size = 8
        self.vocab = vocab
        self.args = args

        self._cdim = 32
        self._wdim = 100
        self._idim = 256

        self._qtdim = 8

        self._bilstm_dim = 128

        self._lstm_layer = 2
        self._mlp_layer = 1
        self._mlp_dim = 128

        self._lstm_dropout = 0.2
        self._mlp_dropout = 0.2

        self._coverage = False

        self.device = device

        self._keyword_lookup = nn.Embedding(num_embeddings=len(self.vocab['keyword']), embedding_dim=self._idim, padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)
        self._columnt_lookup = nn.Embedding(num_embeddings=len(self.vocab['columnt']), embedding_dim=self._idim, padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)
        self._colkey_lookup = nn.Embedding(num_embeddings=len(self.vocab['col_key']), embedding_dim=self._idim, padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)

        self._decoder_lookup = nn.Embedding(num_embeddings=1, embedding_dim=self._idim, max_norm=None, scale_grad_by_freq=False, sparse=False)

        q_input_dim = self._wdim + self._qtdim * 5 + self._bilstm_dim * 2
        self._bert_model = BertModel.from_pretrained("bert-base-uncased")
        self._bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self._bert_w_project = nn.Linear(self._bert_model.pooler.dense.in_features, self._bilstm_dim * 2, bias=False)
        self._bert_c_project = nn.Linear(self._bert_model.pooler.dense.in_features, self._bilstm_dim * 2, bias=False)

        self.c_attn_w = MatchAttn(self._bilstm_dim * 2, identity=True)
        self.w_attn_c = MatchAttn(self._bilstm_dim * 2, identity=True)

        self._q_bilstm_2 = Encoder_rnn(self.args, self._bilstm_dim * 4, self._bilstm_dim)
        self._c_bilstm_2 = Encoder_rnn(self.args, self._bilstm_dim * 4, self._bilstm_dim)

        self.decode_c_attn = MatchAttn(self._bilstm_dim * 2, identity=True)
        self.decode_w_attn = MultiHeadedAttention(1, self._bilstm_dim * 2, dropout=0., coverage=self._coverage)

        self._h_lstm = nn.LSTM(self._idim, self._bilstm_dim * 2, 2, dropout=self._lstm_dropout)

        self._type_final = torch.nn.Sequential(
            nn.Linear(self._bilstm_dim * 6, self._mlp_dim),
            nn.Dropout(self._mlp_dropout),
            nn.ReLU(),
            nn.Linear(self._mlp_dim, len(self.vocab['type']))
        )


        self._col_type_final = torch.nn.Sequential(
            nn.Linear(self._bilstm_dim * 6, self._mlp_dim),
            nn.Dropout(self._mlp_dropout),
            nn.ReLU(),
            nn.Linear(self._mlp_dim, 3)
        )


        self._keyword_final = torch.nn.Sequential(
            nn.Linear(self._bilstm_dim * 6, self._mlp_dim),
            nn.Dropout(self._mlp_dropout),
            nn.ReLU(),
            nn.Linear(self._mlp_dim, len(self.vocab['keyword']))
        )
        self._columnt_final = nn.Linear(self._bilstm_dim * 4, self._bilstm_dim * 2)
        self._colkey_linear = nn.Linear(self._bilstm_dim * 4, self._bilstm_dim * 2)
        self._col_w_key_linear = nn.Linear(self._bilstm_dim * 4, self._bilstm_dim * 2)
        self._col_w_key_two_linear = nn.Linear(self._bilstm_dim * 6, self._bilstm_dim * 2)

        self._column_final = torch.nn.Sequential(
            nn.Linear(self._bilstm_dim * 8, self._mlp_dim),
            nn.Dropout(self._mlp_dropout),
            nn.ReLU(),
            nn.Linear(self._mlp_dim, 1)
        )

        if self.args.aux_col:
            self._aux_col = MatchAttn(self._bilstm_dim * 2, self._bilstm_dim * 2)

        self._column_biaffine = MatchAttn(self._bilstm_dim * 6, self._bilstm_dim * 2)
        self._valbeg_biaffine = MatchAttn(self._bilstm_dim * 6, self._bilstm_dim * 2)
        self._valend_biaffine = MatchAttn(self._bilstm_dim * 6, self._bilstm_dim * 2)

        self._column2i = torch.nn.Sequential(
            nn.Linear(self._bilstm_dim * 2, self._mlp_dim),
            nn.Dropout(self._mlp_dropout),
            nn.ReLU(),
            nn.Linear(self._mlp_dim, self._idim)
        )


        #MultiLayerPerceptron([self._bilstm_dim] + [self._mlp_dim] * self._mlp_layer + [self._idim], dy.rectify, self._model)
        self._val2i = torch.nn.Sequential(
            nn.Linear(self._bilstm_dim * 4, self._mlp_dim),
            nn.Dropout(self._mlp_dropout),
            nn.ReLU(),
            nn.Linear(self._mlp_dim, self._idim)
        )

        self.epoch = 0

    def _bert_features(self, batch, isTrain=False):
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
        loss = list()

        wvecs_pre = wvecs
        cvecs_pre = cvecs

        wvecs = torch.cat((wvecs, wcontext), 2)
        cvecs = torch.cat((cvecs, ccontext), 2)

        w_len = batch['word_mask'].data.eq(0).long().sum(1).numpy().tolist()
        wvecs = self._q_bilstm_2(wvecs, w_len)
        c_len = batch['col_mask'].data.eq(0).long().sum(1).numpy().tolist()
        cvecs = self._c_bilstm_2(cvecs, c_len)

        return wvecs_pre, cvecs_pre, wvecs, cvecs, loss



    def forward(self, batch, isTrain=True, gold_decode=False):
        wvecs_pre, cvecs_pre, wvecs, cvecs, loss = self._bert_features(batch, isTrain=isTrain)
        sqls = batch['sqls']
        instances = batch['instances']


        zeros = torch.zeros(1, self._bilstm_dim * 2, device=self.device)
        criterion = nn.NLLLoss(ignore_index=-1, reduction="sum")
        bce_loss = nn.BCEWithLogitsLoss()

        if isTrain:
            for i in range(len(batch['sqls'])):
                # Column prediction component

                decoder_input = self._decoder_lookup(torch.tensor([0], device=self.device))
                hidden = None

                for ystep, (ttype, value, span) in enumerate(sqls[i]):


                    hvec, c_hidden = self._h_lstm(decoder_input.unsqueeze(1), hidden)
                    hidden = c_hidden


                    val = wvecs[i].unsqueeze(0)
                    if self.args.gold_attn:
                        w_context = self.decode_w_attn(hvec, val, val, batch['word_mask'].to(self.device)[i], gold_attn=att[ystep])
                    else:
                        w_context = self.decode_w_attn(hvec, val, val, batch['word_mask'].to(self.device)[i])
                    c_context, c_score = self.decode_c_attn(hvec, cvecs[i].unsqueeze(0), batch['col_mask'].to(self.device)[i].unsqueeze(0))


                    hvec = torch.cat((hvec, w_context, c_context), -1)

                    # decide current prediction token type
                    potential = self._type_final(hvec).squeeze(0)

                    loss.append(criterion(F.log_softmax(potential, dim=1), torch.tensor([self.vocab['type'][ttype]], device=self.device)))

                    # Keyword prediction
                    if ttype == "Keyword":
                        potential = self._keyword_final(hvec).squeeze(0)
                        loss.append(criterion(F.log_softmax(potential, dim=1), torch.tensor([self.vocab['keyword'][value[0]]], device=self.device)))
                        k_value = torch.tensor([self.vocab['keyword'][value[0]]], device=self.device)

                        ivec = self._keyword_lookup(k_value)

                    # Column prediction
                    elif ttype == "Column":
                        potential = self._col_type_final(hvec).squeeze(0)
                        if span == 'Text':
                            lb = 0 
                            candidates = instances[i]['text_candidates']
                        elif span == 'Date':
                            lb = 1
                            candidates = instances[i]['date_candidates']
                        elif span == 'Number':
                            lb = 2
                            candidates = instances[i]['num_candidates']
                        loss.append(criterion(F.log_softmax(potential, dim=1), torch.tensor([lb], device=self.device)))

                        if lb == 0 or lb == 1:
                            for cand in candidates:
                                
                                s = cand.split("_")
                                col = int(s[0][1:]) - 1
                                col_vec = cvecs[i][col].unsqueeze(0)
                                
                                columnt = '_'.join(s[1:])
                                columnt_vec = self._columnt_lookup( torch.tensor(self.vocab['columnt'][columnt], device=self.device)).unsqueeze(0)
                                col_vec = torch.cat((col_vec, columnt_vec), 1)
                                col_vec = self._columnt_final(col_vec)
                                col_sc = self._column_final(torch.cat((hvec.squeeze(0), col_vec), 1).squeeze(0))
                                if cand == value[0]:
                                    col_lb = 1.0
                                else:
                                    col_lb = 0.0
                                loss.append(bce_loss(col_sc, torch.tensor([col_lb], device=self.device)))
                                ivec = self._column2i(col_vec)

                        if lb == 2:
                            for cand in candidates:
                                items = cand.split('<SPC>')
                                if cand == '<count> ( * )':
                                    col_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<count>'], device=self.device)).unsqueeze(0)
                                    col_vec1 = self._colkey_lookup( torch.tensor(self.vocab['col_key']['*'], device=self.device)).unsqueeze(0)
                                    col_vec = self._colkey_linear(torch.cat((col_vec, col_vec1), 1))
                                    col_sc = self._column_final(torch.cat((hvec.squeeze(0), col_vec), 1).squeeze(0))
                                    if value[0] == 'COUNT ( * )':
                                        col_lb = 1.0
                                    else:
                                        col_lb = 0.0
                                    loss.append(bce_loss(col_sc, torch.tensor([col_lb], device=self.device)))
                                    ivec = self._column2i(col_vec)
                                    
                                


                                elif len(items) == 1:
                                    s = cand.split("_")
                                    col = int(s[0][1:]) - 1
                                    col_vec = cvecs[i][col].unsqueeze(0)
                                    
                                    columnt = '_'.join(s[1:])
                                    columnt_vec = self._columnt_lookup( torch.tensor(self.vocab['columnt'][columnt], device=self.device)).unsqueeze(0)
                                    col_vec = self._columnt_final(torch.cat((col_vec, columnt_vec), 1))
                                    col_sc = self._column_final(torch.cat((hvec.squeeze(0), col_vec), 1).squeeze(0))
                                    if cand == value[0]:
                                        col_lb = 1.0
                                    else:
                                        col_lb = 0.0
                                    loss.append(bce_loss(col_sc, torch.tensor([col_lb], device=self.device)))
                                    ivec = self._column2i(col_vec)
                                elif len(items) == 2:
                                    if items[0] == '<avg> ( <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<avg>'], device=self.device)).unsqueeze(0)
                                    elif items[0] == '<max> ( <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<max>'], device=self.device)).unsqueeze(0)
                                    elif items[0] == '<min> ( <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<min>'], device=self.device)).unsqueeze(0)
                                    elif items[0] == '<sum> ( <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<sum>'], device=self.device)).unsqueeze(0)
                                    elif items[0] == '<count> ( <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<count>'], device=self.device)).unsqueeze(0)
                                    elif items[0] == '<length> ( <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<length>'], device=self.device)).unsqueeze(0)
                                    elif items[0] == '<count> (  <distinct> <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<count>'], device=self.device)).unsqueeze(0)
                                        col_key_vec1 = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<distinct>'], device=self.device)).unsqueeze(0)
                                        col_key_vec = self._colkey_linear(torch.cat((col_key_vec, col_key_vec1), 1))
                                    
                                    s = items[1].split("_")
                                    col = int(s[0][1:]) - 1
                                    col_vec = cvecs[i][col].unsqueeze(0)
                                    
                                    columnt = '_'.join(s[1:])
                                    columnt_vec = self._columnt_lookup( torch.tensor(self.vocab['columnt'][columnt], device=self.device)).unsqueeze(0)
                                    col_vec = self._columnt_final(torch.cat((col_vec, columnt_vec), 1))
                                    col_vec = self._col_w_key_linear(torch.cat((col_vec, col_key_vec), 1))
                                    col_sc = self._column_final(torch.cat((hvec.squeeze(0), col_vec), 1).squeeze(0))

                                    if items[0] == value[0] and len(value[1]) > 0 and items[1] == value[1][0]:
                                        col_lb = 1.0
                                    else:
                                        col_lb = 0.0
                                    loss.append(bce_loss(col_sc, torch.tensor([col_lb], device=self.device)))
                                    ivec = self._column2i(col_vec)

                                elif len(items) == 3:
                                    if items[0] == '<col> + <col>':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['+'], device=self.device)).unsqueeze(0)
                                    elif items[0] == '<col> - <col>':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['-'], device=self.device)).unsqueeze(0)
                                    elif items[0] == 'julianday ( <col> ) - julianday ( <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<julianday> - <julianday>'], device=self.device)).unsqueeze(0)
                                    elif items[0] == '<max> ( <col> ) - <min> ( <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<max> - <min>'], device=self.device)).unsqueeze(0)
                                    
                                    
                                    s = items[1].split("_")
                                    col = int(s[0][1:]) - 1
                                    col_vec = cvecs[i][col].unsqueeze(0)
                                    
                                    columnt = '_'.join(s[1:])
                                    columnt_vec = self._columnt_lookup( torch.tensor(self.vocab['columnt'][columnt], device=self.device)).unsqueeze(0)
                                    col_vec = self._columnt_final(torch.cat((col_vec, columnt_vec), 1))

                                    s = items[2].split("_")
                                    col = int(s[0][1:]) - 1
                                    col_vec1 = cvecs[i][col].unsqueeze(0)
                                    
                                    columnt = '_'.join(s[1:])
                                    columnt_vec = self._columnt_lookup( torch.tensor(self.vocab['columnt'][columnt], device=self.device)).unsqueeze(0)
                                    col_vec1 = self._columnt_final(torch.cat((col_vec1, columnt_vec), 1))

                                    col_vec = self._col_w_key_two_linear(torch.cat((col_vec, col_vec1, col_key_vec), 1))
                                    col_sc = self._column_final(torch.cat((hvec.squeeze(0), col_vec), 1).squeeze(0))

                                    if items[0] == value[0] and len(value[1]) > 1 and items[1] == value[1][0] and items[2] == value[1][1]:
                                        col_lb = 1.0
                                    elif items[0] == value[0] and len(value[1]) > 1 and items[1] == value[1][1] and items[2] == value[1][0]:
                                        col_lb = 1.0
                                    else:
                                        col_lb = 0.0
                                    loss.append(bce_loss(col_sc, torch.tensor([col_lb], device=self.device)))

                        
                                    ivec = self._column2i(col_vec)
                        

                    # Literal Prediction
                    else:
                        potential = self._valbeg_biaffine(hvec, wvecs[i].unsqueeze(0), batch['word_mask'].to(self.device)[i].unsqueeze(0), is_score=True).squeeze(0)
                        loss.append(criterion(F.log_softmax(potential, dim=1), torch.tensor([span[0]], device=self.device)))

                        if len(span) > 1:
                            potential = self._valend_biaffine(hvec, wvecs[i].unsqueeze(0), batch['word_mask'].to(self.device)[i].unsqueeze(0), is_score=True).squeeze(0)
                            loss.append(criterion(F.log_softmax(potential, dim=1), torch.tensor([span[1]], device=self.device)))

                            ivec = self._val2i(torch.cat((wvecs[i][span[0]].unsqueeze(0), wvecs[i][span[1]].unsqueeze(0)), 1))
                        else:
                            ivec = self._val2i(torch.cat((wvecs[i][span[0]].unsqueeze(0), wvecs[i][span[0]].unsqueeze(0)), 1))

                    decoder_input = ivec

            return loss
        else:
            logprobs = []
            pred_data = list()
            for i in range(len(batch['sqls'])):
                query = []
                types = []
                ii = 0
                json_file = "../data/tables/json/{}.json".format(instances[i]["tbl"])
                with open(json_file, "r") as f:
                    table = json.load(f)
                
                new_contents = list()
                new_contents.append(table['contents'][0])
                new_contents.append(table['contents'][1])
                for jj, ele in enumerate(table['contents'][2:]):
                    if jj in instances[i]['column_indexs']:
                        new_contents.append(ele)
                table['contents'] = new_contents
                
                cells = get_cells(table)
                if len(cells) == 0:
                    continue
                hidden = None
                decoder_input = self._decoder_lookup(torch.tensor([0], device=self.device))

                if gold_decode:
                    gold_sql = batch['sqls'][i]


                while True:
                    if gold_decode:
                        gold_ttype, gold_value, gold_span = gold_sql[ii]
                    logprob = 0.
                    ii += 1
                    if ii> 100:
                        break

                    hvec, c_hidden = self._h_lstm(decoder_input.unsqueeze(1), hidden)
                    hidden = c_hidden


                    val = wvecs[i].unsqueeze(0)
                    if self.args.gold_attn:
                        if ii - 1 < att.shape[0]:
                            w_context = self.decode_w_attn(hvec, val, val, batch['word_mask'].to(self.device)[i], gold_attn=att[ii - 1])
                        else:
                            w_context = self.decode_w_attn(hvec, val, val, batch['word_mask'].to(self.device)[i], gold_attn=att[-1])
                    else:
                        w_context = self.decode_w_attn(hvec, val, val, batch['word_mask'].to(self.device)[i])
                    c_context, c_score = self.decode_c_attn(hvec, cvecs[i].unsqueeze(0), batch['col_mask'].to(self.device)[i].unsqueeze(0))
                    w_score = self.decode_w_attn.attn.squeeze(0).squeeze(1)[0]



                    hvec = torch.cat((hvec, w_context, c_context), -1)

                    potential = F.log_softmax(self._type_final(hvec).squeeze(0), dim=1).squeeze(0).data.cpu().numpy()

                    if not instances[i]["has_number"]:
                        potential[self.vocab['type']["Literal.Number"]] = -np.inf
                    if len(cells) == 0 :
                        potential[self.vocab['type']["Literal.String"]] = -np.inf

                    if gold_decode:
                        choice = self.vocab['type'][gold_ttype]
                    else:
                        choice = np.argmax(potential)

                    ttype = list(self.vocab['type'].keys())[choice]
                    types.append(ttype)
                    logprob += potential[choice]

                    if ttype == "Keyword":

                        potential = F.log_softmax(self._keyword_final(hvec).squeeze(0), dim=1).squeeze(0).data.cpu().numpy()
                        if gold_decode:
                            iv = self.vocab['keyword'][gold_value]
                        else:
                            iv = np.argmax(potential)

                        value = list(self.vocab['keyword'].keys())[iv]
                        #print(value)

                        logprob += potential[iv]

                        if value == "<EOS>":
                            logprobs.append(logprob)
                            break

                        query.append(value)

                        k_value = torch.tensor([self.vocab['keyword'][value]], device=self.device)
                        ivec = self._keyword_lookup(k_value)

                    elif ttype == "Column":

                        
                        potential = F.log_softmax(self._col_type_final(hvec).squeeze(0), dim=1).squeeze(0).data.cpu().numpy()
                        if len(instances[i]['date_candidates']) == 0:
                            potential[1] =  -np.inf
                        if len(instances[i]['text_candidates']) == 0:
                            potential[0] =  -np.inf
                        
                        #print(potential)
                        iv = np.argmax(potential)
                        #if iv == 0:
                        #    iv = 2
                        if iv == 0:
                            candidates = instances[i]['text_candidates']
                        elif iv == 1:
                            candidates = instances[i]['date_candidates']
                        elif iv == 2:
                            candidates = instances[i]['num_candidates']

                        all_cols = list()
                        all_col_scores = list()
                        all_col_vecs = list()

                        if iv == 0 or iv == 1:
                            for cand in candidates:
                                
                                s = cand.split("_")
                                col = int(s[0][1:]) - 1
                                col_vec = cvecs[i][col].unsqueeze(0)
                                
                                columnt = '_'.join(s[1:])
                                if columnt not in self.vocab['columnt']:
                                    continue


                                columnt_vec = self._columnt_lookup(torch.tensor(self.vocab['columnt'][columnt], device=self.device)).unsqueeze(0)
                                col_vec = torch.cat((col_vec, columnt_vec), 1)
                                col_vec = self._columnt_final(col_vec)
                                col_sc = self._column_final(torch.cat((hvec.squeeze(0), col_vec), 1).squeeze(0))
                                all_cols.append(cand)
                                all_col_scores.append(col_sc.data.cpu().numpy()[0])
                                all_col_vecs.append(col_vec)

                        if iv == 2:
                            for cand in candidates:
                                items = cand.split('<SPC>')
                                if cand == '<count> ( * )':
                                    col_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<count>'], device=self.device)).unsqueeze(0)
                                    col_vec1 = self._colkey_lookup( torch.tensor(self.vocab['col_key']['*'], device=self.device)).unsqueeze(0)
                                    col_vec = self._colkey_linear(torch.cat((col_vec, col_vec1), 1))
                                    col_sc = self._column_final(torch.cat((hvec.squeeze(0), col_vec), 1).squeeze(0))
                                    all_cols.append(cand)
                                    all_col_scores.append(col_sc.data.cpu().numpy()[0])
                                    all_col_vecs.append(col_vec)
                                    
                                    
                                elif len(items) == 1:
                                    s = cand.split("_")
                                    col = int(s[0][1:]) - 1
                                    col_vec = cvecs[i][col].unsqueeze(0)
                                    
                                    columnt = '_'.join(s[1:])
                                    if columnt not in self.vocab['columnt']:
                                        continue
                                    columnt_vec = self._columnt_lookup( torch.tensor(self.vocab['columnt'][columnt], device=self.device)).unsqueeze(0)
                                    col_vec = self._columnt_final(torch.cat((col_vec, columnt_vec), 1))

                                    col_sc = self._column_final(torch.cat((hvec.squeeze(0), col_vec), 1).squeeze(0))
                                    all_cols.append(cand)
                                    all_col_scores.append(col_sc.data.cpu().numpy()[0])
                                    all_col_vecs.append(col_vec)
                                elif len(items) == 2:
                                    if items[0] == '<avg> ( <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<avg>'], device=self.device)).unsqueeze(0)
                                    elif items[0] == '<max> ( <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<max>'], device=self.device)).unsqueeze(0)
                                    elif items[0] == '<min> ( <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<min>'], device=self.device)).unsqueeze(0)
                                    elif items[0] == '<sum> ( <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<sum>'], device=self.device)).unsqueeze(0)
                                    elif items[0] == '<count> ( <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<count>'], device=self.device)).unsqueeze(0)
                                    elif items[0] == '<length> ( <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<length>'], device=self.device)).unsqueeze(0)
                                    elif items[0] == '<count> (  <distinct> <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<count>'], device=self.device)).unsqueeze(0)
                                        col_key_vec1 = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<distinct>'], device=self.device)).unsqueeze(0)
                                        col_key_vec = self._colkey_linear(torch.cat((col_key_vec, col_key_vec1), 1))
                                    
                                    s = items[1].split("_")
                                    col = int(s[0][1:]) - 1
                                    col_vec = cvecs[i][col].unsqueeze(0)
                                    
                                    columnt = '_'.join(s[1:])
                                    if columnt not in self.vocab['columnt']:
                                        continue
                                    columnt_vec = self._columnt_lookup( torch.tensor(self.vocab['columnt'][columnt], device=self.device)).unsqueeze(0)
                                    col_vec = self._columnt_final(torch.cat((col_vec, columnt_vec), 1))
                                    col_vec = self._col_w_key_linear(torch.cat((col_vec, col_key_vec), 1))
                                    col_sc = self._column_final(torch.cat((hvec.squeeze(0), col_vec), 1).squeeze(0))

                                    all_cols.append(cand)
                                    all_col_scores.append(col_sc.data.cpu().numpy()[0])
                                    all_col_vecs.append(col_vec)
                                elif len(items) == 3:
                                    if items[0] == '<col> + <col>':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['+'], device=self.device)).unsqueeze(0)
                                    elif items[0] == '<col> - <col>':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['-'], device=self.device)).unsqueeze(0)
                                    elif items[0] == 'julianday ( <col> ) - julianday ( <col> )':
                                        col_key_vec = self._colkey_lookup( torch.tensor(self.vocab['col_key']['<julianday> - <julianday>'], device=self.device)).unsqueeze(0)
                                    
                                    s = items[1].split("_")
                                    col = int(s[0][1:]) - 1
                                    col_vec = cvecs[i][col].unsqueeze(0)
                                    
                                    columnt = '_'.join(s[1:])
                                    if columnt not in self.vocab['columnt']:
                                        continue
                                    columnt_vec = self._columnt_lookup( torch.tensor(self.vocab['columnt'][columnt], device=self.device)).unsqueeze(0)
                                    col_vec = self._columnt_final(torch.cat((col_vec, columnt_vec), 1))

                                    s = items[2].split("_")
                                    col = int(s[0][1:]) - 1
                                    col_vec1 = cvecs[i][col].unsqueeze(0)
                                    
                                    columnt = '_'.join(s[1:])
                                    if columnt not in self.vocab['columnt']:
                                        continue
                                    columnt_vec = self._columnt_lookup( torch.tensor(self.vocab['columnt'][columnt], device=self.device)).unsqueeze(0)
                                    col_vec1 = self._columnt_final(torch.cat((col_vec1, columnt_vec), 1))

                                    col_vec = self._col_w_key_two_linear(torch.cat((col_vec, col_vec1, col_key_vec), 1))
                                    col_sc = self._column_final(torch.cat((hvec.squeeze(0), col_vec), 1).squeeze(0))

                                    all_cols.append(cand)
                                    all_col_scores.append(col_sc.data.cpu().numpy()[0])
                                    all_col_vecs.append(col_vec)

                        ### todo
                        if len(all_col_scores) == 0:
                            continue
                        max_index = all_col_scores.index(max(all_col_scores))

                        ivec = self._column2i(all_col_vecs[max_index])
                        types.append('Column')
                        all_items = candidates[max_index].split('<SPC>')
                        if len(all_items) == 1:
                            query.append(candidates[max_index])
                        elif len(all_items) == 2:
                            full_str = all_items[0].replace('<col>', all_items[1])
                            for ele in full_str.split(' '):
                                query.append(ele)
                        elif len(all_items) == 3:
                            if all_items[0] == '<col> - <col>':
                                query.append(all_items[1])
                                query.append('-')
                                query.append(all_items[2])
                            if all_items[0] == '<col> + <col>':
                                query.append(all_items[1])
                                query.append('+')
                                query.append(all_items[2])
                            if all_items[0] == 'julianday ( <col> ) - julianday ( <col> )':
                                query.append('julianday')
                                query.append('(')
                                query.append(all_items[1])
                                query.append(')')
                                query.append('-')
                                query.append('julianday')
                                query.append('(')
                                query.append(all_items[2])
                                query.append(')')
                            if all_items[0] == '<max> ( <col> ) - <min> ( <col> )':
                                query.append('<max>')
                                query.append('(')
                                query.append(all_items[1])
                                query.append(')')
                                query.append('-')
                                query.append('<min>')
                                query.append('(')
                                query.append(all_items[2])
                                query.append(')')



                    else:

                        if ttype == "Literal.String":
                            potential = F.log_softmax(self._valbeg_biaffine(hvec, wvecs[i].unsqueeze(0), batch['word_mask'].to(self.device)[i].unsqueeze(0), is_score=True).squeeze(0), dim=1).squeeze(0).data.cpu().numpy()
                            if gold_decode:
                                span_beg = gold_span[0]
                            else:
                                span_beg = np.argmax(potential)
                            logprob += potential[span_beg]

                            potential = F.log_softmax(self._valend_biaffine(hvec, wvecs[i].unsqueeze(0), batch['word_mask'].to(self.device)[i].unsqueeze(0), is_score=True).squeeze(0), dim=1).squeeze(0).data.cpu().numpy()
                            if gold_decode:
                                span_end = gold_span[1]
                            else:
                                span_end = np.argmax(potential[span_beg:]) + span_beg
                            logprob += potential[span_end]

                            if len(query) >= 2 and query[-1] == "=" and types[-3] == "Column":
                                col, literal = best_match(cells, " ".join(instances[i]["nl"][span_beg:span_end+1]), query[-2])
                            else:
                                col, literal = best_match(cells, " ".join(instances[i]["nl"][span_beg:span_end+1]))
                            

                            col_id = int(col.split('_')[0][1:]) - 1
                            new_col = 'c' + str(instances[i]['column_indexs'].index(col_id) + 1)
                            if len(col.split('_')) > 1:
                                new_col += '_'
                                new_col += '_'.join(col.split('_')[1:])
                            col = new_col
                            
                            query.append("{}".format(repr(literal)))
                            

                            # postprocessing, fix the col = val mismatch
                            if len(query) >= 3 and query[-2] == "=" and types[-3] == "Column":
                                query[-3] = col

                            if len(query) >= 4 and query[-2] == "(" and query[-3] == "in" and types[-4] == "Column":
                                query[-4] = col
                        else:
                            potential = F.log_softmax(self._valbeg_biaffine(hvec, wvecs[i].unsqueeze(0), batch['word_mask'].to(self.device)[i].unsqueeze(0), is_score=True).squeeze(0), dim=1).squeeze(0).data.cpu().numpy()
                            for j, n in enumerate(instances[i]["numbers"]):
                                if n is None:
                                    potential[j] = -np.inf

                            if gold_decode:
                                span_beg = gold_span[0]
                            else:
                                span_beg = np.argmax(potential)
                            logprob += potential[span_beg]
                            span_end = span_beg
                            query.append("{}".format(parse_number(instances[i]["nl"][span_beg])))
                        ivec = self._val2i(torch.cat((wvecs[i][span_beg].unsqueeze(0), wvecs[i][span_end].unsqueeze(0)), 1))

                    decoder_input = ivec
                    logprobs.append(logprob)



                _query = query
                types = " ".join(types)
                query = ' '.join(query).lower()
                #print(query)
                
                query =query.replace('<count>', 'count')
                query =query.replace('<max>', 'max')
                query =query.replace('<min>', 'min')
                query =query.replace('<*>', '*')
                query =query.replace('<distinct>', 'distinct')
                query =query.replace('<avg>', 'avg')
                query =query.replace('<sum>', 'sum')
                query =query.replace('<length>', 'length')
                tgt_query = []
                for x in instances[i]['first_stage']:
                    if x[0] == 'Keyword':
                        tgt_query.append(x[1][0].lower())

                    elif x[0] != 'Column':
                        tgt_query.append(str(x[1]))
                    else:
                        ele = x[1]
                        if len(ele[1]) == 0:
                            tgt_query.append(ele[0].lower())
                        elif len(ele[1]) == 1:
                            app_str = ele[0].replace('<col>', ele[1][0])
                            tgt_query.append(app_str)
                        else:
                            print('wrong')

                tgt_query = ' '.join(tgt_query)

                tgt_query = tgt_query.replace('<count>', 'count')
                tgt_query = tgt_query.replace('<max>', 'max')
                tgt_query = tgt_query.replace('<min>', 'min')
                tgt_query = tgt_query.replace('<*>', '*')
                tgt_query = tgt_query.replace('<distinct>', 'distinct')
                tgt_query = tgt_query.replace('<avg>', 'avg')
                tgt_query = tgt_query.replace('<sum>', 'sum')
                tgt_query = tgt_query.replace('<length>', 'length')


                pred_data.append({
                'table_id': instances[i]["tbl"],
                'result': [
                        {
                        'sql': query,
                        'sql_type': types,
                        'id': instances[i]["nt"],
                        'tgt': tgt_query,
                        'old_tgt':" ".join([x[1] for x in instances[i].get("sql", [])]),
                        'nl': ' '.join(instances[i]['nl'])
                        }
                    ]
                })

            return pred_data, logprobs


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


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None, gate=None, gold_attn=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask.bool(), -float('inf'))
    p_attn = F.softmax(scores, dim = -1)

    if gate:
        gated = torch.sigmoid(gate(query))
        p_attn = p_attn * gated

    if dropout is not None:
        p_attn = dropout(p_attn)

    if gold_attn is not None:
        length = gold_attn.shape[0]
        first_head = p_attn[:, 0:1, :, :]
        rest = p_attn[:, 1:, :, :]

        new_first = torch.zeros_like(first_head)
        new_first[0][0][0][:length] = torch.Tensor(gold_attn[:length])
        p_attn = torch.cat((new_first, rest), 1)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, gate=False, coverage=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        if gate:
            self.gate = nn.Linear(self.d_k, 1)
        else:
            self.gate = None

        if coverage:
            self.linear_cover = nn.Linear(1, self.d_k)
        else:
            self.linear_cover = None

    def forward(self, query, key, value, mask=None, coverage=None, gold_attn=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            # mask = mask.unsqueeze(1)
            pass
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # coverage
        if coverage is not None:
            key += self.linear_cover(coverage.unsqueeze(-1))

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout, gold_attn=gold_attn)
                                 # dropout=self.dropout, gate=self.gate)
        #print(self.attn)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)
