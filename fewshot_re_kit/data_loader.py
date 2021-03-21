'''
This script is modified based on FewRel Toolkit
FewRel Toolkit Source Code: https://github.com/thunlp/FewRel
'''

import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import sys

class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate=0s, root, kg_enhance=None):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file:", path," does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.kg_enhance = kg_enhance

    def __getraw__(self, item):
        if self.kg_enhance:
            word, pos1, pos2, mask, head_cnpt_emb, tail_cnpt_emb = self.encoder.tokenize(item['tokens'],
                item['h'][2][0],
                item['t'][2][0],
                item['h'][1],
                item['t'][1])
            return word, pos1, pos2, mask, head_cnpt_emb, tail_cnpt_emb
        else:
            word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0],)
            return word, pos1, pos2, mask
        

    def __additem__(self, d, word, pos1, pos2, mask, head_cnpt_emb, tail_cnpt_emb, cls_name):
        if self.kg_enhance:
            d['word'].append(word)
            d['pos1'].append(pos1)
            d['pos2'].append(pos2)
            d['mask'].append(mask)
            d['head_cnpt_emb'].append(head_cnpt_emb)
            d['tail_cnpt_emb'].append(tail_cnpt_emb)
            d['cls_name'].append(cls_name)
        else:
            d['word'].append(word)
            d['pos1'].append(pos1)
            d['pos2'].append(pos2)
            d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        if self.kg_enhance:
            support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'head_cnpt_emb': [], 'tail_cnpt_emb': [], 'cls_name': []}
            query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'head_cnpt_emb': [], 'tail_cnpt_emb': [], 'cls_name': []}
        else:
            support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
            query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,  
            self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                if self.kg_enhance:
                    word, pos1, pos2, mask, head_cnpt_emb, tail_cnpt_emb = self.__getraw__(
                            self.json_data[class_name][j])
                    word = torch.tensor(word).long()
                    pos1 = torch.tensor(pos1).long()
                    pos2 = torch.tensor(pos2).long()
                    mask = torch.tensor(mask).long()
                    head_cnpt_emb = head_cnpt_emb.detach()
                    tail_cnpt_emb = tail_cnpt_emb.detach()

                    cls_name = torch.tensor(i)
                    if count < self.K:
                        self.__additem__(support_set, word, pos1, pos2, mask, head_cnpt_emb, tail_cnpt_emb, cls_name)
                    else:
                        self.__additem__(query_set, word, pos1, pos2, mask, head_cnpt_emb, tail_cnpt_emb, cls_name)
                else:
                    word, pos1, pos2, mask = self.__getraw__(
                            self.json_data[class_name][j])
                    word = torch.tensor(word).long()
                    pos1 = torch.tensor(pos1).long()
                    pos2 = torch.tensor(pos2).long()
                    mask = torch.tensor(mask).long()

                    if count < self.K:
                        self.__additem__(support_set, word, pos1, pos2, mask)
                    else:
                        self.__additem__(query_set, word, pos1, pos2, mask)
                count += 1

            query_label += [i] * self.Q


        return support_set, query_set, query_label
    
    def __len__(self):
        return 1000000000

def collate_fn_desc(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'head_cnpt_emb': [], 'tail_cnpt_emb': [], 'cls_name': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'head_cnpt_emb': [], 'tail_cnpt_emb': [], 'cls_name': []}

    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)

    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)

    return batch_support, batch_query, batch_label

def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label


def get_loader(name, encoder, N, K, Q, batch_size, 
        num_workers=0, collate_fn=collate_fn, na_rate=0, root='./data', kg_enhance=None, use_ernie=False):
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root, kg_enhance)
    if kg_enhance is not None:
        collate_fn=collate_fn_desc

    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

