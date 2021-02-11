import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class MyProto(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, dropout=0,  word_embedding_dim=50, feature_size=0, hidden_size=230, kg_dim=256, add_cnpt_node=False, kg_enrich=None):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.drop = nn.Dropout()
        self.word_embedding_dim =  word_embedding_dim
        self.hidden_size = hidden_size
        self.feature_size =  feature_size
        self.kg_dim = kg_dim
        self.kg_enrich = kg_enrich
        self.add_cnpt_node = add_cnpt_node

        self.input_dim_hs = self.word_embedding_dim + self.feature_size
        self.input_dim_fc = self.word_embedding_dim + self.feature_size

        if add_cnpt_node:
            self.input_dim_hs = self.word_embedding_dim + self.feature_size * 2 
            self.input_dim_fc = self.word_embedding_dim + self.feature_size * 2 + self.kg_dim*2
        else:
            self.input_dim_hs = self.word_embedding_dim + self.feature_size * 2
            self.input_dim_fc = self.word_embedding_dim + self.feature_size * 2


        self.linear_trans_desc = nn.Linear(self.input_dim_hs, self.hidden_size, bias=False)
        self.linear_trans_cnpt = nn.Linear(self.hidden_size + self.kg_dim * 2 , self.hidden_size, bias=True)


    # L2
    def __dist__(self, x, y, dim):
        return -(torch.pow(x - y, 2)).sum(dim) #((B,QN,N))

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3) # S:(B,1,N,D)，Q:(B,QN,1,D)

    def forward(self, support, query, N, K, total_Q, do_self_att=False):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''

        support_emb, support_cnpt_emb = self.sentence_encoder(inputs=support) # (B * N * K, max_len, D), where D is the hidden size
        query_emb, query_cnpt_emb = self.sentence_encoder(inputs=query) # (B * total_Q, max_len, D)

        support_emb = self.linear_trans_desc(support_emb)  # (B * N * K,max_len, H)
        query_emb = self.linear_trans_desc(query_emb) # (B * total_Q,max_len, H)
        

        if self.kg_enrich is not None:
            if self.add_cnpt_node:
                s_head_cnpt_emb = torch.squeeze(support_cnpt_emb[:,:1])
                s_tail_cnpt_emb = torch.squeeze(support_cnpt_emb[:,1:])

                q_head_cnpt_emb = torch.squeeze(query_cnpt_emb[:,:1])
                q_tail_cnpt_emb = torch.squeeze(query_cnpt_emb[:,1:])

                if q_head_cnpt_emb.dim() != query_emb.dim():
                    q_head_cnpt_emb = torch.unsqueeze(q_head_cnpt_emb, 0)
                    q_tail_cnpt_emb = torch.unsqueeze(q_tail_cnpt_emb, 0)

                support_head_tail = torch.cat((support_emb, s_head_cnpt_emb, s_tail_cnpt_emb),-1)
                support_emb = self.linear_trans_cnpt(support_head_tail) # (B * N * K, H)

                query_head_tail = torch.cat((query_emb, q_head_cnpt_emb, q_tail_cnpt_emb),-1)
                query_emb = self.linear_trans_cnpt(query_head_tail) # (B * N * K, H)


        support = self.drop(support_emb)
        query = self.drop(query_emb)

        support = support.view(-1, N, K, self.hidden_size) # (B, N, K, D)
        query = query.view(-1, total_Q, self.hidden_size) # (B, total_Q, D)

        B = support.size(0) # Batch size
         
        # Prototypical Networks 
        support = torch.mean(support, 2) # Calculate prototype for each class，(B, N, H)
        logits = self.__batch_dist__(support, query) # (B, total_Q, H)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N+1), 1)
        
        return logits, pred
    

