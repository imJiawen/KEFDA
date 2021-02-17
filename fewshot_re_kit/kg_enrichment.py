import sys
sys.path.append('..')
import fewshot_re_kit
import networkx as nx
import json
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os.path

class KG_Enrichment():
    def __init__(self, kg_dict="./fewshot_re_kit/kg_toolkit/wiki_onto", multi=False, pad_num=0, kg_dim=256, kg_encoder="distmult"):
        if kg_encoder == "distmult":
            self.emb_file = kg_dict + "/kg_embedding/distmult_wiki_umls_sn.embedding"
        elif kg_encoder == "analogy":
            self.emb_file = kg_dict + "/kg_embedding/analogy_wiki_umls_sn.embedding"
        elif kg_encoder == "transe":
            self.emb_file = kg_dict + "/kg_embedding/transe_wiki_umls_sn.embedding"
        elif kg_encoder == "rotate":
            self.emb_file = kg_dict + "/kg_embedding/rotate_wiki_umls_sn.embedding"
        else:
            print("invalid kg encoder name")
            sys.exit(0)

        self.ent_embeds = self.load_kg_emb(self.emb_file, dim=kg_dim)

        self.triple_file = kg_dict + "/triple2id.txt"
        self.entity2id_file = kg_dict + "/entity2id.txt"
        self.relation2id_file = kg_dict + "/relation2id.txt"
        self.ent2cnpt_file = kg_dict + "/ent2cnpt.txt"
        self.ent2path_file = kg_dict + "/ent2path.dict"
        self.cnpt2id_file = kg_dict + "/concept2id.txt"
        self.cnpt_id2desc_file = kg_dict + "/cnpt_id2def_fea.txt"

        self.desc_ent2id_file = kg_dict + "/desc_feature/umls_wiki_cui2id.txt"
        self.id2desc_file = kg_dict + "/desc_feature/umls_wiki_id2fea.txt"


        self.multi = multi

        self.G = self.build_graph()
        self.ent2id, self.id2ent = self.build_dict(self.entity2id_file)
        self.rel2id, self.id2rel = self.build_dict(self.relation2id_file)
        self.cnpt2id, self.id2cnpt = self.build_dict(self.cnpt2id_file)
        self.ent2cnpt = self.build_dict(self.ent2cnpt_file, id_dict=False)

        # description features for entity
        self.desc_ent2id, self.desc_id2ent = self.build_dict(self.desc_ent2id_file, spliter=' ')
        self.id2desc_fea, self.fea_dim, tot_num = self.read_id2fea_dict(self.id2desc_file)

        # description features for concept
        self.cnpt_id2desc_fea, self.cnpt_fea_dim, _ = self.read_id2fea_dict(self.cnpt_id2desc_file)
        self.cnpt_desc_emb = self.load_emb_for_dict(self.cnpt_id2desc_fea, self.cnpt_fea_dim, len(self.cnpt2id))

        if os.path.isfile(self.ent2path_file):
            self.ent2cnpt_path = json.load(open(self.ent2path_file, 'r', encoding='utf8'))
            print("loaded dict from ", self.ent2path_file, " done, ", str(len(self.ent2cnpt_path)), " items loaded.")
        else:
            self.ent2cnpt_path = {}

        self.wiki_search_cnpt = fewshot_re_kit.kg_toolkit.wikidata_entlink.wiki_search_cnpt(kg_folder=kg_dict)
        self.pad_num = pad_num


    def load_emb_for_dict(self, emb_dict, dim, tot_num):
        init_weight = torch.full((tot_num, dim), -float('inf'))
        for idx in emb_dict:
            init_weight[idx] = emb_dict[idx]

        emb = nn.Embedding.from_pretrained(init_weight)
        print("transfer from dict to nn.Embedding done")
        return emb
        
    def read_id2fea_dict(self, id2fea_path):
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
        return id2fea_dict, fea_dim, count_num


    def get_cnpt_emb(self, ent=None, cnpt_set=None, input_cnptset=False):
        cnpt = None

        if input_cnptset:
            if cnpt_set is not None and len(cnpt_set) != 0:
                cnpt = cnpt_set
            else:
                return None
        elif ent in self.ent2cnpt:
            cnpt = self.ent2cnpt[ent]
        else:
            return None

        if cnpt is None:
            return None

        cnpt_id_list = []
        for i in cnpt:
            if i in self.ent2id:
                cnpt_id_list.append(self.ent2id[i])

        if cnpt_id_list != []:
            cnpt_emb =  self.ent_embeds(torch.LongTensor(cnpt_id_list))
        else:
            return None

        return cnpt_emb

    def build_graph(self):
        node_list, graph = self.read_triple_as_graph(self.triple_file)
        G = nx.Graph()
        if self.multi:
            G = nx.MultiGraph()

        G.add_nodes_from(node_list)
        G.add_edges_from(graph)

        return G

    def read_triple_as_graph(self,data_file):
        # e1 e2 r
        graph = []
        node_list = []
        with open(data_file,'r', encoding='utf8') as f:
            for line in f:
                line = line.strip('\n')
                line = line.split(' ')

                if len(line) != 3:
                    continue

                if line[0] not in node_list:
                    node_list.append(line[0])
                if line[1] not in node_list:
                    node_list.append(line[1])

                graph.append((int(line[0]), int(line[1]), {"route": int(line[2])}))

        print(str(len(graph)), " triples loaded from file ", data_file)

        return node_list, graph


    def build_dict(self, file_path, id_dict=True, spliter='\t'):
        item2id_dict = {}
        id2item_dict = {}
        ent2cnpt_dict = {}
        with open(file_path,'r', encoding='utf8') as f:
            for line in f:
                line = line.strip('\n')
                line = line.split(spliter)

                if id_dict:
                    if len(line) != 2:
                        continue
                    
                    item2id_dict.update({line[0]:int(line[1])})
                    id2item_dict.update({int(line[1]):line[0]})

                else:
                    if len(line) >= 2:
                        ent2cnpt_dict.update({line[0]:list(line[1:])})

        if id_dict:
            print("loaded dict from ", file_path, " done, ", str(len(item2id_dict)), " items loaded.")
            return item2id_dict, id2item_dict
        else:
            print("loaded dict from ", file_path, " done, ", str(len(ent2cnpt_dict)), " items loaded.")
            return ent2cnpt_dict


    def load_kg_emb(self, emb_file, dim=256, ent_tot=6536, rel_tot=82):
        with open(emb_file,'r',encoding='utf8') as f:
            data = json.load(f)
            ent_embeds = nn.Embedding(ent_tot, dim)

            ent_pretrained_weight = np.array(data["ent_embeddings.weight"])
            ent_embeds.weight.data.copy_(torch.from_numpy(ent_pretrained_weight))

            print("loaded embedding from ", emb_file, " done")
            return ent_embeds


    def cal_neighbour_mean(self, neibour_path):
        neibour_path = self.ent_embeds(torch.LongTensor(neibour_path))
        return torch.mean(neibour_path, 0)

    def str2list(self, str_data, spliter="\t"):
        list_str_data = str_data.split(spliter)
        list_int_data = [int(i) for i in list_str_data]
        return list_int_data

    def list2str(self, list_data, spliter="\t", keep_ht=True):
        res = []
        for item in list_data:
            if keep_ht:
                res.append("\t".join(map(str, item)))
            elif len(item) > 2:
                res.append("\t".join(map(str, item[1:-1])))
                
        return res






