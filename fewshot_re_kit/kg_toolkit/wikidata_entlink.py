import json
from tqdm import tqdm
import time
import sys
import re
import requests

def read_list_file(data_file, data=None):
    if not data:
        data = set()
    with open(data_file,'r', encoding='utf8') as f:
        for line in f:
            line = line.strip('\n')
            data.add(line)

    print(str(len(data)), " loaded in file ", data_file)
    return data

def read_triple_file_to_node(data_file, data=None):
    if not data:
        data = set()
    with open(data_file,'r', encoding='utf8') as f:
        for line in f:
            line = line.strip('\n')
            line = line.split('\t')
            data.add(line[1])

    print(str(len(data)), " loaded in file ", data_file)
    return data

def save_list_file(save_path, data_set):
    if isinstance(save_path,str):
        fout = open(save_path, 'w', encoding='utf8')
    else:
        fout = save_path

    for i in data_set:
        fout.write(str(i) + '\n')

    print(str(len(data_set)), " data saved to ", save_path)

def save_dict_file(save_path, data_dict):
    # input: {key: {set}}
    fout = open(save_path, 'w', encoding='utf8')
    for i in data_dict:
        fout.write(i + '\t' + '\t'.join([j for j in data_dict[i]]) + '\n')
    print(str(len(data_dict)), " dict saved to ", save_path)


class wiki_search_cnpt():
    def __init__(self, kg_folder):
        self.entity_uri = "Q35120"  #entity
        self.is_instance = "P31" 
        self.sub_class = "P279"

        self.url = 'https://query.wikidata.org/sparql'
        self.wdt = "http://www.wikidata.org/prop/direct/"
        self.wd = "http://www.wikidata.org/entity/"

        self.concept_file = kg_folder + "/wiki_onto.node"
        self.concept_set = read_list_file(self.concept_file)
        self.third_concept_file = kg_folder + "/wiki_onto_third.node"
        self.third_concept_set = read_list_file(self.third_concept_file)
        self.third_concept_in_meta_file = kg_folder + "/wiki_onto_third_in_meta.node"
        self.third_concept_in_meta_set = read_list_file(self.third_concept_in_meta_file)


    def wiki_request(self, query_string):
        cnpts = set()
        #print("get....")
        r = requests.get(self.url, params = {'format': 'json', 'query': query_string})
        #print("get finish")
        while r.status_code == 429:
            print("retry...")
            #time.sleep(int(r.headers["Retry-After"]))
            time.sleep(2)
            r = requests.get(self.url, params = {'format': 'json', 'query': query_string})

        data = r.json()
        for item in data['results']['bindings']:
            i = item['WDid']['value'].strip(self.wd)
            if i != '':
                cnpts.add(i)

        return cnpts

    def search_concept(self, ent):
        res_cnpts = set()
        
        if ent in self.concept_set:
            return {ent}
        
        query_string = "SELECT $WDid WHERE {wd:"+ ent +" wdt:P31/wdt:P279* ?WDid }"
        #print("search_concept")
        cnpts = self.wiki_request(query_string)
        #print(cnpts)

        if len(cnpts) == 0:
            query_string = "SELECT $WDid WHERE {wd:"+ ent +" wdt:P279* ?WDid }"
            cnpts = self.wiki_request(query_string)
        #print(cnpts)

        res_cnpts = cnpts & self.third_concept_in_meta_set
        #print(res_cnpts)

        if len(res_cnpts) == 0:
            res_cnpts = cnpts & self.third_concept_set

        if len(res_cnpts) != 0:
            return res_cnpts
        else:
            return None
        



#if __name__ == "__main__":
    
    #onto_json_file = "./data/wiki_onto/wiki_onto_3rd_metarel2.json"
    #onto_json_file = "./data/wiki_onto/wiki_onto_3rd_tree.json"
    #save_file = "./data/wiki_onto/wiki_onto_third_in_meta.node"
    #onto_triple = "./data/wiki_onto/wiki_onto.triple"
    #onto_nodes = "./data/wiki_onto/wiki_onto.node"
    #wikidata_create_ontology().onto_convert(onto_json_file,save_file)
    #wikidata_create_ontology().collect_cnpt(onto_triple,onto_nodes)
    #res = wiki_search_cnpt().find_concept("Q1598759")

#onto_json_file = "./data/wiki_onto/wiki_onto_fir_layer.json"
#save_file = "./data/wiki_onto/wiki_onto1.triple"
#wikidata_create_ontology().onto_convert(onto_json_file,save_file)

#print(wiki_search_cnpt().search_concept("Q2288135"))