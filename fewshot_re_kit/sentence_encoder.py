'''
This script is modified based on FewRel Toolkit
FewRel Toolkit Source Code: https://github.com/thunlp/FewRel
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import sys
from torch import optim
from . import network
import transformers
from . import knowledge_bert


def read_id2fea_dict(id2fea_path):
    id2fea_dict = {}
    count_num = 0
    with open(id2fea_path,'r',encoding="utf8") as f:
        for line in f:
            line = line.strip('\n')
            line = line.split(' ')

            if len(line) <= 2:
                continue

            id2fea_dict.update({int(line[0]):torch.Tensor([float(i) for i in line[1:]])})
            count_num += 1
            fea_dim = len(line[1:])

    print("Read id2fea dict done, ", count_num," nodes loaded")
    return id2fea_dict, fea_dim

def read_ent2id_dict(ent2id_path):
    ent2id_dict = {}
    id2ent_dict = {}
    count_num = 0
    with open(ent2id_path,'r',encoding="utf8") as f:
        for line in f:
            line = line.strip('\n')
            line = line.split(' ')

            ent2id_dict.update({line[0]:int(line[1])})
            id2ent_dict.update({int(line[1]):line[0]})
            count_num += 1

    print("Read ent2id dict done, ", count_num," nodes loaded")
    return ent2id_dict, id2ent_dict


class BERTSentenceEncoder(nn.Module):
    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.bert = transformers.BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, inputs):
        _, x = self.bert(inputs['word'], attention_mask=inputs['mask'])
        return x
        
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask



class BERTSentenceEncoder_EntDesc(nn.Module):
    def __init__(self, pretrain_path, max_length, id2desc_path, ent2id_path=None, kg_enrich=None, add_cnpt_node=False, kg_emb_dim=256, path_num=300, path_node=8, pad_num=0): 
        nn.Module.__init__(self)
        self.bert = transformers.BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        self.id2desc_path = id2desc_path
        self.id2fea_dict, self.fea_dim = read_id2fea_dict(id2desc_path)
        if ent2id_path:
            self.ent2id_dict, self.id2ent_dict = read_ent2id_dict(ent2id_path)
            
        self.kg_enrich = kg_enrich
        self.kg_emb_dim = kg_emb_dim
        self.pad_num = pad_num
        self.path_num = path_num
        self.path_node = path_node
        self.add_cnpt_node = add_cnpt_node

    def get_concepts_mean_emb(self, ent):
        cnpt_embs = self.kg_enrich.get_cnpt_emb(ent)
        if cnpt_embs is not None:
            cnpt_emb_mean = torch.mean(cnpt_embs, 0).reshape(1,self.kg_emb_dim)
        else:
            cnpt_emb_mean = torch.full((1,self.kg_emb_dim), float(self.pad_num))

        return cnpt_emb_mean


    def forward(self, inputs): 
        outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])

        sent_emb = None
        cnpt_emb = None

        for i in range(len(outputs[1])):
            #get the sum of concepts emb & the concept path
            if self.kg_enrich is not None:
                head_cnpt_emb = inputs['head_cnpt_emb'][i]
                tail_cnpt_emb = inputs['tail_cnpt_emb'][i]
                cnpt_path_emb = inputs['cnpt_path_emb'][i]
                head_id = inputs['pos1'][i][2].item()
                tail_id = inputs['pos2'][i][2].item()
                
                cnpt = torch.cat((head_cnpt_emb,tail_cnpt_emb),-1)

                if cnpt_emb is not None:
                    cnpt_emb = torch.cat((cnpt_emb,cnpt),0)
                else:
                    cnpt_emb = cnpt

            
            head_fea = torch.zeros(self.fea_dim,dtype=torch.float)
            tail_fea = torch.zeros(self.fea_dim,dtype=torch.float)

            if torch.cuda.is_available():
                head_fea = head_fea.cuda()
                tail_fea = tail_fea.cuda()

            if head_id in self.id2fea_dict:
                head_fea = self.id2fea_dict[head_id]
                if torch.cuda.is_available():
                    head_fea = head_fea.cuda()

            if tail_id in self.id2fea_dict:
                tail_fea = self.id2fea_dict[tail_id]
                if torch.cuda.is_available():
                    tail_fea = tail_fea.cuda()

            tmp_sent = torch.cat((outputs[1][i], head_fea, tail_fea), -1) # (s,h,t) (D+H+H)

            if sent_emb is not None:
                sent_emb = torch.cat((sent_emb, torch.unsqueeze(tmp_sent,0)),0)
            else:
                sent_emb = torch.unsqueeze(tmp_sent,0)

        
        if torch.cuda.is_available():
            sent_emb = sent_emb.cuda()
            if cnpt_emb is not None:
                cnpt_emb = cnpt_emb.cuda()

        return sent_emb, cnpt_emb

    def tokenize(self, raw_tokens, pos_head, pos_tail, head_cui, tail_cui):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        head_pos1_in_index = 1
        head_pos2_in_index = 1
        tail_pos1_in_index = 1
        tail_pos2_in_index = 1

        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                head_pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                tail_pos1_in_index = len(tokens)

            tokens += self.tokenizer.tokenize(token)

            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
                head_pos2_in_index = len(tokens)
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
                tail_pos2_in_index = len(tokens)
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]
 
        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - head_pos1_in_index + self.max_length
            pos2[i] = i - tail_pos1_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        head_pos1_in_index = min(self.max_length, head_pos1_in_index)
        head_pos2_in_index = min(self.max_length, head_pos2_in_index)
        tail_pos1_in_index = min(self.max_length, tail_pos1_in_index)
        tail_pos2_in_index = min(self.max_length, tail_pos2_in_index)

        head_id = -1
        tail_id = -1
        if head_cui in self.ent2id_dict:
            head_id = self.ent2id_dict[head_cui]

        if tail_cui in self.ent2id_dict:
            tail_id = self.ent2id_dict[tail_cui]

        if self.kg_enrich is not None and self.add_cnpt_node:
            head_cnpt = self.get_concepts_mean_emb(head_cui) #(1, self.kg_emb_dim)
            tail_cnpt = self.get_concepts_mean_emb(tail_cui) #(1, self.kg_emb_dim)
        else:
            head_cnpt = torch.full((1,self.kg_emb_dim), float(self.pad_num))
            tail_cnpt = torch.full((1,self.kg_emb_dim), float(self.pad_num))

        return indexed_tokens, [head_pos1_in_index-1, head_pos2_in_index-1, head_id] , [tail_pos1_in_index-1, tail_pos2_in_index-1,tail_id], mask, head_cnpt, tail_cnpt


class BERTSentenceEncoder_cnpt_id(nn.Module):
    #get sent with cnpt id
    def get_concepts_mean_emb(self, ent):
        cnpt_embs = self.kg_enrich.get_cnpt_emb(ent)
        if cnpt_embs is not None:
            cnpt_emb_mean = torch.mean(cnpt_embs, 0).reshape(1,self.kg_emb_dim)
        else:
            cnpt_emb_mean = torch.full((1,self.kg_emb_dim), float(self.pad_num))

        return cnpt_emb_mean

    def __init__(self, pretrain_path, max_length, kg_enrich=None, kg_emb_dim=256, pad_num=0): 
        nn.Module.__init__(self)

        self.bert = transformers.BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        self.id2fea_dict, self.fea_dim = kg_enrich.id2desc_fea, kg_enrich.fea_dim
        self.ent2id_dict, self.id2ent_dict = kg_enrich.desc_ent2id, kg_enrich.desc_id2ent
            
        self.kg_enrich = kg_enrich
        self.kg_emb_dim = kg_emb_dim
        self.pad_num = pad_num

    def forward(self, N, K, inputs): 
        outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])

        origin_sent_emb = None
        sent_emb = None
        cnpt_emb = None
        head = []
        pair_list = []
        pair_list_batch = []

        for i in range(len(outputs[1])):
            #get the sum of concepts emb & the concept path
            if self.kg_enrich is not None:
                head_cnpt_emb = inputs['head_cnpt_emb'][i]
                tail_cnpt_emb = inputs['tail_cnpt_emb'][i]
                head_id = inputs['pos1'][i][2].item()
                tail_id = inputs['pos2'][i][2].item()
                
                head_cnpt = None
                tail_cnpt = None
                if head_id in self.id2ent_dict:
                    head_ent = self.id2ent_dict[head_id]
                    if head_ent in self.kg_enrich.ent2cnpt:
                        head_cnpt = self.kg_enrich.ent2cnpt[head_ent]

                if tail_id in self.id2ent_dict:
                    tail_ent = self.id2ent_dict[tail_id]
                    if tail_ent in self.kg_enrich.ent2cnpt:
                        tail_cnpt = self.kg_enrich.ent2cnpt[tail_ent]

                if head_cnpt is not None and tail_cnpt is not None:
                    for h in head_cnpt:
                        for t in tail_cnpt:
                            if h in self.kg_enrich.ent2id and t in self.kg_enrich.ent2id:
                                pair_list.append((self.kg_enrich.ent2id[h], self.kg_enrich.ent2id[t]))

                if (i+1) % K == 0:
                    pair_list_batch.append(pair_list)
                    pair_list = []

                
                cnpt = torch.cat((head_cnpt_emb,tail_cnpt_emb),-1)

                if cnpt_emb is not None:
                    cnpt_emb = torch.cat((cnpt_emb,cnpt),0)
                else:
                    cnpt_emb = cnpt

            head_fea = torch.zeros(self.fea_dim,dtype=torch.float)
            tail_fea = torch.zeros(self.fea_dim,dtype=torch.float)

            if torch.cuda.is_available():
                head_fea = head_fea.cuda()
                tail_fea = tail_fea.cuda()

            if head_id in self.id2fea_dict:
                head_fea = self.id2fea_dict[head_id]
                if torch.cuda.is_available():
                    head_fea = head_fea.cuda()

            if tail_id in self.id2fea_dict:
                tail_fea = self.id2fea_dict[tail_id]
                if torch.cuda.is_available():
                    tail_fea = tail_fea.cuda()

            tmp_sent = torch.cat((outputs[1][i], head_fea, tail_fea), -1) # (s,h,t) (D+H+H)

            if sent_emb is not None:
                sent_emb = torch.cat((sent_emb, torch.unsqueeze(tmp_sent,0)),0)
            else:
                sent_emb = torch.unsqueeze(tmp_sent,0)

            if origin_sent_emb is not None:
                origin_sent_emb = torch.cat((origin_sent_emb, torch.unsqueeze(outputs[1][i],0)),0)
            else:
                origin_sent_emb = torch.unsqueeze(outputs[1][i],0)
        
        if torch.cuda.is_available():
            sent_emb = sent_emb.cuda()
            if cnpt_emb is not None:
                cnpt_emb = cnpt_emb.cuda()
            if origin_sent_emb is not None:
                origin_sent_emb = origin_sent_emb.cuda()


        return sent_emb, cnpt_emb, pair_list_batch, origin_sent_emb


    def tokenize(self, raw_tokens, pos_head, pos_tail, head_cui, tail_cui, rel_id=None):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        head_pos1_in_index = 1
        head_pos2_in_index = 1
        tail_pos1_in_index = 1
        tail_pos2_in_index = 1

        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                head_pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                tail_pos1_in_index = len(tokens)

            tokens += self.tokenizer.tokenize(token)

            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
                head_pos2_in_index = len(tokens)
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
                tail_pos2_in_index = len(tokens)
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]
 
        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - head_pos1_in_index + self.max_length
            pos2[i] = i - tail_pos1_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        head_pos1_in_index = min(self.max_length, head_pos1_in_index)
        head_pos2_in_index = min(self.max_length, head_pos2_in_index)
        tail_pos1_in_index = min(self.max_length, tail_pos1_in_index)
        tail_pos2_in_index = min(self.max_length, tail_pos2_in_index)

        head_id = -1
        tail_id = -1
        if head_cui in self.ent2id_dict:
            head_id = self.ent2id_dict[head_cui]

        if tail_cui in self.ent2id_dict:
            tail_id = self.ent2id_dict[tail_cui]

        if self.kg_enrich:
            head_cnpt = self.get_concepts_mean_emb(head_cui) #(1, self.kg_emb_dim)
            tail_cnpt = self.get_concepts_mean_emb(tail_cui) #(1, self.kg_emb_dim)
        else:
            head_cnpt = torch.full((1,self.kg_emb_dim), float(self.pad_num))
            tail_cnpt = torch.full((1,self.kg_emb_dim), float(self.pad_num))

        return indexed_tokens, [head_pos1_in_index-1, head_pos2_in_index-1, head_id] , [tail_pos1_in_index-1, tail_pos2_in_index-1,tail_id], mask, head_cnpt, tail_cnpt

