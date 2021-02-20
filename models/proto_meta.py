import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
import random
from copy import deepcopy

def neg_sampler(n_pair_list, sample_size, samehead=False):
    neg_samples = []
    pair_list = tuple(n_pair_list)
    if not samehead:
        while len(neg_samples) < sample_size:
            h = random.choice(range(6535))
            t = random.choice(range(6535))
            if (h, t) not in n_pair_list and (t, h) not in n_pair_list:
                neg_samples.append((h, t))
    else: 
        # pick the same head entity
        n = 0
        while n < sample_size:
            h = pair_list[n][0]
            t = random.choice(range(6535))
            if (h, t) not in n_pair_list and (t, h) not in n_pair_list:
                neg_samples.append((h, t))
                n += 1

    return list(neg_samples)

class RelationMetaLearner_FNN(nn.Module):
    # using FNN to learn relation meta
    def __init__(self, embed_size=256, num_hidden=256, out_size=256, dropout_p=0.5):
        super(RelationMetaLearner_FNN, self).__init__()
        self.embed_size = embed_size
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden, num_hidden)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden, out_size)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, x):
        x = self.rel_fc1(x) #(K, kg_dim)
        x = self.rel_fc2(x) #(K, kg_dim)
        x = self.rel_fc3(x) #(K, kg_dim)

        return x


class SentenceEnhancement(nn.Module):
    def __init__(self, hidden_size, feature_dim, kg_dim):
        super(SentenceEnhancement, self).__init__()
        self.sent_in_dim = hidden_size + feature_dim*2
        self.add_feature_dim =  hidden_size + kg_dim*2
        self.linear_combine_desc = nn.Linear(self.sent_in_dim, hidden_size, bias=True)
        self.linear_combine_cnpt = nn.Linear(self.add_feature_dim, hidden_size, bias=True)

    def forward(self, x, features):
        x1 = self.linear_combine_desc(x)  # (B * N * K,max_len, H)
        x2 = self.linear_combine_cnpt(torch.cat((x1, features), -1))
        return x1, x2

class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, r, t): # (K, H), (H,H), (K,H)
        if r.dim() == 1:  #in: (D), (k, D)  out:  (k, 1, D)
            r = r.unsqueeze(0)
        h = h.view(1, -1, h.shape[-1])
        t = t.view(1, -1, t.shape[-1])
        r = r.view(r.shape[0], -1, r.shape[-1])

        score = (h * r) * t  #(N, K, D)
        score = torch.sum(score, -1)
        return -score #(N.K)

class CnptAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CnptAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key):
        '''
        query: sent_emb (1, D)
        key: [(k, D), (k,D)]
        value: (k, kg_dim, kg_dim)
        '''
        num = key[0].shape[0]
        query = query.view(1, query.shape[-1]).expand(num, -1)

        # L2
        h_score = -(torch.pow(query - key[0], 2)).sum(-1).unsqueeze(-1)
        t_score = -(torch.pow(query - key[1], 2)).sum(-1).unsqueeze(-1)
        h_t_score = torch.cat((h_score, t_score), -1) #(k,2)

        score, _ = torch.max(h_t_score, -1) #(k, 1)
        score = self.softmax(score)
        return score.squeeze()  #(k)


class RelationMetaLearner(nn.Module):
    def __init__(self, N, K, beta, kg_enrich, meta_learn_method='bilinear',kg_dim=256, margin=1):
        super(RelationMetaLearner, self).__init__()
        self.kg_enrich = kg_enrich
        self.embedding_learner = EmbeddingLearner()
        self.relation_meta_learner = RelationMetaLearner_FNN(embed_size=kg_dim, num_hidden=kg_dim, out_size=kg_dim)
        self.y = torch.Tensor([1])
        self.loss_func = nn.MarginRankingLoss(margin)
        self.beta = beta
        self.b = nn.Parameter(torch.zeros(1))
        hidden_size = 768
        self.cnpt_attention = CnptAttention(hidden_size*3, hidden_size)

        if torch.cuda.is_available():
            self.y = self.y.cuda()
            self.b = self.b.cuda()


    def forward(self, n, pair_list, cnpt_att=False, query=None, samehead=False):
        n_pair_list = pair_list[n]
        sample_size = len(n_pair_list)

        if sample_size == 0:
            return None

        head_id_list = []
        tail_id_list = []

        for pair in n_pair_list:
            if pair[0] in self.kg_enrich.cnpt_id2desc_fea and pair[1] in self.kg_enrich.cnpt_id2desc_fea:
                head_id_list.append(pair[0])
                tail_id_list.append(pair[1])

        if len(head_id_list) == 0:
            cnpt_att = False

        pos_head_id_list = [i[0] for i in n_pair_list]
        pos_tail_id_list = [i[1] for i in n_pair_list]

        neg_list = neg_sampler(n_pair_list, sample_size, samehead=samehead)

        pos_head_emb = self.kg_enrich.ent_embeds(torch.LongTensor(pos_head_id_list)) #(k, 256)
        pos_tail_emb = self.kg_enrich.ent_embeds(torch.LongTensor(pos_tail_id_list))

        neg_head_emb = self.kg_enrich.ent_embeds(torch.LongTensor([i[0] for i in neg_list])) #(k, 256)
        neg_tail_emb = self.kg_enrich.ent_embeds(torch.LongTensor([i[1] for i in neg_list]))

        if torch.cuda.is_available():
            pos_head_emb = pos_head_emb.cuda()
            pos_tail_emb = pos_tail_emb.cuda()
            neg_head_emb = neg_head_emb.cuda()
            neg_tail_emb = neg_tail_emb.cuda()

        pos_head_tail_emb = torch.cat((pos_head_emb, pos_tail_emb), -1) #(k, 512)
        
        # using the head-tail embedding to learn a rel meta
        rel_meta = self.relation_meta_learner(pos_head_tail_emb) # in: (K, 512) outL: (K, 256)
        
        if not cnpt_att:
            rel_meta = torch.mean(rel_meta, 0) #(256)

        else:
            assert query is not None
            head_desc_fea = self.kg_enrich.cnpt_desc_emb(torch.LongTensor(pos_head_id_list)).cuda() #(k, D)
            tail_desc_fea = self.kg_enrich.cnpt_desc_emb(torch.LongTensor(pos_tail_id_list)).cuda() #(k, D)

            if torch.cuda.is_available():
                head_desc_fea = head_desc_fea.cuda()
                tail_desc_fea = tail_desc_fea.cuda()

            key = [head_desc_fea, tail_desc_fea]
            score = self.cnpt_attention(query, key) #(k)
            score = score.unsqueeze(-1)  #[k, 1]

            rel_meta = rel_meta * score #[k,D]
            rel_meta = torch.sum(rel_meta, 0) #[D]

        rel_meta.retain_grad()

        rel_meta_s = rel_meta
        p_score = self.embedding_learner(pos_head_emb, rel_meta_s, pos_head_emb)
        n_score = self.embedding_learner(neg_head_emb, rel_meta_s, neg_tail_emb)
        torch.cuda.empty_cache()
        self.zero_grad()
        loss = self.loss_func(p_score, n_score, self.y)
        loss.backward(retain_graph=True)
        grad_meta = rel_meta.grad
        rel_meta_q = rel_meta - self.beta*grad_meta

        self.zero_grad()

        del loss

        del pos_head_emb
        del pos_tail_emb
        del neg_head_emb
        del neg_tail_emb

        return rel_meta_q


class ProtoMeta(fewshot_re_kit.framework.FewShotREModel):
    def __init__(self, sentence_encoder, N, K, dot=False,  dropout=0,  word_embedding_dim=50, feature_size=0, hidden_size=230, kg_dim=256, kg_enrich=None, lambda_para=0.5, beta=1, lambda_para=0.7, samehead=False, cnpt_att=False):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.N = N
        self.K = K
        self.drop = nn.Dropout()
        self.dot = dot
        self.word_embedding_dim =  word_embedding_dim
        self.hidden_size = hidden_size
        self.feature_size =  feature_size
        self.kg_dim = kg_dim
        self.margin = 1
        self.kg_enrich = kg_enrich
        self.beta = beta

        self.sent_in_dim = self.hidden_size + self.feature_size*2
        self.add_feature_dim =  self.hidden_size + self.kg_dim*2

        self.sentence_enhance = SentenceEnhancement(self.hidden_size, self.feature_size, self.kg_dim)
        self.relation_meta_learner = RelationMetaLearner( N, K, self.beta, self.kg_enrich, kg_dim=self.kg_dim)

        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.l2loss = nn.MSELoss()
        self.softmax = nn.Softmax(dim=-1)
        self.y = torch.Tensor([1])
        self.zero_r = torch.full((1, self.kg_dim), -float(1))
        self.lambda_para = lambda_para
        self.lambda_para = self.lambda_para
        self.batch_norm = nn.BatchNorm1d(N)
        self.samehead = samehead
        self.cnpt_att = cnpt_att

        if torch.cuda.is_available():
            self.y = self.y.cuda()
            self.zero_r = self.zero_r.cuda()
    #L2
    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim) #((B,QN,N))

    def __batch_dist__(self, S, Q): # (B, N, H), (B, Q, H)
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3) # S:(B,1,N,D)，Q:(B,QN,1,D)


    def forward(self, support, query, label, N, K, total_Q, is_eval=False, is_test=False):
        #knowledge enhance module
        support_emb, support_cnpt_emb, support_cnpt_pair_ids, support_origin_sent_emb = self.sentence_encoder(N, K, support) # (B * N * K, D), (B*N, 2)
        query_emb, query_cnpt_emb, query_cnpt_pair_ids, query_origin_sent_emb = self.sentence_encoder(N, 1, query) # (B * total_Q, D)


        support_emb = support_emb.view(-1, K, support_emb.shape[-1])
        support_cnpt_emb = support_cnpt_emb.view(-1, K, support_cnpt_emb.shape[-1])

        query_emb = query_emb.view(-1, total_Q, query_emb.shape[-1])
        query_cnpt_emb = query_cnpt_emb.view(-1, total_Q, query_cnpt_emb.shape[-1])

        support_desc_features, support_all_features = self.sentence_enhance(support_emb, support_cnpt_emb)
        query_desc_features, query_all_features = self.sentence_enhance(query_emb, query_cnpt_emb)

        support_proto = torch.mean(support_all_features, dim=-2)
        support_proto = support_proto.view(-1, N, support_proto.shape[-1])

        query_desc_features = query_desc_features.view(total_Q, query_desc_features.shape[-1])

        query_emb = query_emb.squeeze()

        # relation meta learner
        triplet_loss = 0
        count = 0
        meta_batch = None
        query_ht_emb_list = []

        cnpt_mask = [1 for _ in range(N)]  
        for n in range(N):
            rel_meta = self.relation_meta_learner(n, support_cnpt_pair_ids, cnpt_att=self.cnpt_att, query=support_origin_sent_emb[n], samehead=self.samehead)  #(1, H,H)

            # determine if there is triple in query
            query_sample_size = len(query_cnpt_pair_ids[n])
            if query_sample_size == 0:
                query_ht_emb_list.append(None)

            else:
                neg_list = neg_sampler(query_cnpt_pair_ids[n], query_sample_size, samehead=self.samehead)

                pos_head_emb = self.kg_enrich.ent_embeds(torch.LongTensor([i[0] for i in query_cnpt_pair_ids[n]])) #(k, 256)
                pos_tail_emb = self.kg_enrich.ent_embeds(torch.LongTensor([i[1] for i in query_cnpt_pair_ids[n]]))

                neg_head_emb = self.kg_enrich.ent_embeds(torch.LongTensor([i[0] for i in neg_list])) #(k, 256)
                neg_tail_emb = self.kg_enrich.ent_embeds(torch.LongTensor([i[1] for i in neg_list]))

                if torch.cuda.is_available():
                    pos_head_emb = pos_head_emb.cuda()
                    pos_tail_emb = pos_tail_emb.cuda()
                    neg_head_emb = neg_head_emb.cuda()
                    neg_tail_emb = neg_tail_emb.cuda()

                query_ht_emb_list.append([pos_head_emb, pos_tail_emb])

            if rel_meta is None:
                cnpt_mask[n] = 0
                if meta_batch is None:
                    meta_batch = self.zero_r
                else:
                    meta_batch = torch.cat((meta_batch, self.zero_r), 0)
            else:
                if meta_batch is None:
                    meta_batch = torch.unsqueeze(rel_meta, 0).detach()
                else:
                    meta_batch = torch.cat((meta_batch, torch.unsqueeze(rel_meta, 0).detach()), 0)

            if query_sample_size == 0 or rel_meta is None:
                continue

            if not is_eval:
                p_score = self.embedding_learner(pos_head_emb, rel_meta, pos_tail_emb) # (K, 1,1)
                n_score = self.embedding_learner(neg_head_emb, rel_meta, neg_tail_emb)

                loss = self.loss_func(p_score, n_score, self.y)
                triplet_loss +=  float(loss)
                count += 1

        self.zero_grad()

        triplet_logit = None
        if not is_eval:
            if count > 0:
                triplet_loss = triplet_loss / count
            
            for n in range(total_Q):
                if query_ht_emb_list[n] is None:
                    triplet_score =  torch.full((1, N), float(0))
                    if torch.cuda.is_available():
                        triplet_score = triplet_score.cuda()
                else:
                    triplet_score = self.embedding_learner(query_ht_emb_list[n][0], meta_batch, query_ht_emb_list[n][1]) #(N, k)
                    triplet_score = triplet_score.sum(-1)  #(N)
                    triplet_score = torch.unsqueeze(triplet_score, 0)  #(1, N)

                if triplet_logit is None:
                    triplet_logit = triplet_score
                else:
                    triplet_logit = torch.cat((triplet_logit, triplet_score), 0)  #((QN,N))

            ce_logits = self.__batch_dist__(support_proto, query_all_features) #((B,QN,N))
            ce_logits = ce_logits.view(-1, N)  #((QN,N))

            # if there is no rel meta, lambda is 1
            cnpt_mask = torch.Tensor(cnpt_mask) #(N)
            if torch.cuda.is_available():
                cnpt_mask = cnpt_mask.cuda()
            weight_ce = torch.add(torch.mul(cnpt_mask, (self.lambda_para-1)), 1).unsqueeze(0).expand(total_Q, -1)
            weight_triplet = torch.mul(cnpt_mask, (1-self.lambda_para)).unsqueeze(0).expand(total_Q, -1)

            ce_pred_logits = ce_logits * weight_ce
            triplet_pred_logit = triplet_logit * weight_triplet
            logits = torch.add(ce_pred_logits, triplet_pred_logit) #((QN,N))
            _, pred = torch.max(logits, 1)
            ce_loss = self.loss(ce_logits, label)  # scalar
            loss = (self.lambda_para * ce_loss) + ((1-self.lambda_para) * triplet_loss)
            return logits, pred, loss

        else:
            with torch.no_grad():
                for n in range(total_Q):
                    if query_ht_emb_list[n] is None:
                        triplet_score =  torch.full((1, N), float(0))
                        if torch.cuda.is_available():
                            triplet_score = triplet_score.cuda()
                    else:
                        triplet_score = self.embedding_learner(query_ht_emb_list[n][0], meta_batch, query_ht_emb_list[n][1]) #(N, k)
                        triplet_score = triplet_score.sum(-1)  #(N)
                        triplet_score = torch.unsqueeze(triplet_score, 0)  #(1, N)

                    if triplet_logit is None:
                        triplet_logit = triplet_score
                    else:
                        triplet_logit = torch.cat((triplet_logit, triplet_score), 0)  #((QN,N))

                ce_logits = self.__batch_dist__(support_proto, query_all_features) #((B,QN,N))
                ce_logits = ce_logits.view(-1, N)  #((QN,N))
                
                # if there is no rel meta, lambda is 1
                cnpt_mask = torch.Tensor(cnpt_mask) #(N)
                if torch.cuda.is_available():
                    cnpt_mask = cnpt_mask.cuda()
                    
                weight_ce = torch.add(torch.mul(cnpt_mask, (self.lambda_para-1)), 1).unsqueeze(0).expand(total_Q, -1)
                weight_triplet = torch.mul(cnpt_mask, (1-self.lambda_para)).unsqueeze(0).expand(total_Q, -1)

                ce_pred_logits = ce_logits * weight_ce
                triplet_pred_logit = triplet_logit * weight_triplet

                logits = torch.add(ce_pred_logits, triplet_pred_logit) #((QN,N))
                _, pred = torch.max(logits, 1)
                return logits, pred


