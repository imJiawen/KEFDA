from quickumls import QuickUMLS
import sys

class umls_entity_linking()
    def __init__(self,def_file,quickumls_fp,threshold=0.7,sim_name="jaccard"):
        self.id2ent = self.load_sn_id(def_file)
        self.quickumls_fp = quickumls_fp
        self.matcher = QuickUMLS(self.quickumls_fp,threshold=threshold,similarity_name=sim_name)

    def load_sn_id(self,def_file):
        id2str_dict = {}
        with open(def_file) as f:
            for line in f:
                line = line.strip()
                line = line.strip('\n')
                line = line.split('|')

                id2str_dict.update({line[1]:{"str":line[2],"def":line[4]}})

        return id2str_dict

    def vote_for_type(self,item_match_list):
        sem_cal ={}
        sem_type = ''
        for i in item_match_list:
            sim = i['similarity']
            for st in i['semtypes']:
                if st not in sem_cal:
                    sem_cal.update({st:sim})
                else:
                    sem_cal[st] = sem_cal[st] + sim
        
        # pick the most possible sem type
        sim_max = 0
        for k, v in sem_cal.items():
            if v > sim_max:
                sem_type = k
                sim_max = v

        return sem_type

    def parse_res(self,match_list):
        res_list = []
        for item in match_list:
            type_id = vote_for_type(item)
            type_str = self.id2ent[type_id]["str"]
            res_list.append({'start':item[0]['start'], 'end': item[0]['end'], 
            'ngram': item[0]['ngram'], 'term': item[0]['term'], 'cui': item[0]['cui'],
            'semtypes':type_str})

        return res_list


    def print_list(self,res):
        for i in res:
            print(i)


    def do_entity_linking(self,text):

        if isinstance(text,list):
            text = ' '.join(text)
        elif isinstance(text,str):
            pass
        else:
            print("not valid data type!")
            sys.exit()

        match_list = self.matcher.match(text, best_match=True, ignore_syntax=False)

        res_list = parse_res(match_list)

        #print_list(res_list)


'''
text = ["loss", "of", "ltf", "expression", "was", "observed", "in", "a", "significantly", "higher", "frequency", "of", "npc", "tissues", "compared", "to", "that", "in", "nontumor", "nasopharyngeal", "epithelial", "tissues", "."]
def_file="./SRDEF"
quickumls_fp = '/mnt/d/CODES/UMLS/quickumls_install'
print(text)
do_entity_linking(text,threshold=0.5,sim_name="cosine")
'''