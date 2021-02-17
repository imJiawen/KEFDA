from fewshot_re_kit.data_loader import get_loader
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import BERTSentenceEncoder, BERTSentenceEncoder_EntDesc, BERTSentenceEncoder_cnpt_id
import fewshot_re_kit
import models
from models.proto import Proto
from models.my_proto import MyProto
from models.rel_meta import RelMeta
from models.rel_meta_simp import RelMeta_Sim
from models.maml import MAML
from models.metaR import MetaR
from models.proto_meta import ProtoMeta
from models.proto_meta_finetune import ProtoMetaFineTune
from models.d import Discriminator
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_wiki',
            help='train file')
    parser.add_argument('--val', default='val_wiki',
            help='val file')
    parser.add_argument('--test', default='test_wiki',
            help='test file')
    parser.add_argument('--adv', default=None,
            help='adv file')
    parser.add_argument('--trainN', default=10, type=int,
            help='N in train')
    parser.add_argument('--N', default=5, type=int,
            help='N way')
    parser.add_argument('--K', default=5, type=int,
            help='K shot')
    parser.add_argument('--Q', default=5, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=30000, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=5000, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int,
           help='val after training how many iters')
    parser.add_argument('--model', default='proto_meta',
            help='model name')
    parser.add_argument('--max_length', default=128, type=int,
           help='max length')
    parser.add_argument('--lr', default=1e-1, type=float,
           help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
           help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float,
           help='dropout rate')
    parser.add_argument('--optim', default='sgd',
           help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=230, type=int,
           help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
           help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
           help='only test')

    # for bert
    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert pre-trained checkpoint')
    parser.add_argument('--word_embedding_dim', default=50, type=int,
           help='word_embedding_dim')
    

    parser.add_argument('--gpu', default=None, type=int,
           help='gpu to be used')


    # for knoeledge graph
    parser.add_argument('--kg_dict', default=None,
           help='the dictionary of knowledge graph')
    parser.add_argument('--add_cnpt_node', action='store_true',
           help='add concept embedding as feature')
    parser.add_argument('--kg_encoder', default="distmult",
           help='the knowledge graph encoder')
    parser.add_argument('--feature_dim', default=0, type=int,
           help='dim of additional features')
    parser.add_argument('--return_cnpt_id', action='store_true',
           help='return concept id list in sentence encoder')

    # hyper-parameter
    parser.add_argument('--lambda_para', default=0.5, type=float,
           help='trade off cross entropy loss and triplet loss')
    parser.add_argument('--alpha', default=0.7, type=float,
           help='combine ratio in inference')
    parser.add_argument('--beta', default=1, type=float,
           help='meta learning rate')
    parser.add_argument('--cnpt_att', action='store_true',
           help='use attention in cnpt aggregation')

    opt = parser.parse_args()
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    max_length = opt.max_length
    kg_enrich = None
    kg_encoder = opt.kg_encoder
    cal_meta_rel = False
    return_cnpt_id = opt.return_cnpt_id
    cnpt_att = opt.cnpt_att
    keep_grad = False
    encoder_name = "bert"
    bert_optim = True

    if kg_encoder == "distmult" or "transe":
        kg_emb_dim = 256
    elif kg_encoder == "analogy" or "rotate":
        kg_emb_dim = 512
    else:
        print("error kg encoder format, please choose from [distmult, analogy, transe, rotate]")
        sys.exit(0)

    if model_name == 'proto_meta':
        batch_size = 1
        cal_meta_rel = True
        return_cnpt_id = True
        keep_grad = True

    #if opt.gpu and torch.cuda.is_available():
    #    torch.cuda.set_device(opt.gpu)
    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("val dataset: {}".format(opt.val))
    print("max_length: {}".format(max_length))
    print("hidden dim: {}".format(opt.hidden_size))
    print("feature dim: {}".format(opt.feature_dim))
    print("kg_dir: {}".format(opt.kg_dict))
    print("kg_dim: {}".format(str(kg_emb_dim)))
    print("use_ernie: {}".format(str(opt.use_ernie)))

    prefix = '-'.join([model_name, opt.train, opt.val, str(N), str(K)])
    prefix = prefix + "-lambda_para-" + str(opt.lambda_para) + "-beta-" + str(opt.beta)

    if opt.kg_dict is not None:
       prefix = prefix + '-' + opt.kg_encoder
       ent_desc = opt.kg_dict + "/desc_feature/umls_wiki_id2fea.txt"
       ent2id_file = opt.kg_dict + "/desc_feature/umls_wiki_cui2id.txt"

    else:
       ent_desc = None
       ent2id_file = None

    if opt.cnpt_att:
       prefix += '-cnpt_att'

    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    print("save_ckpt_path: {}".format(ckpt))

    if opt.kg_dict is not None:
        kg_enrich = fewshot_re_kit.kg_enrichment.KG_Enrichment(kg_dict=opt.kg_dict, kg_dim=kg_emb_dim, kg_encoder=kg_encoder)
        print("load kg_enrich done")
    

    pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'

    if return_cnpt_id:
        sentence_encoder = BERTSentenceEncoder_cnpt_id(
                pretrain_ckpt,
                max_length,
                kg_enrich=kg_enrich,
                kg_emb_dim=kg_emb_dim)
    elif kg_enrich is not None:
        sentence_encoder = BERTSentenceEncoder_EntDesc(
                pretrain_ckpt,
                max_length,
                ent_desc,
                ent2id_path=ent2id_file,
                kg_enrich=kg_enrich,
                kg_emb_dim=kg_emb_dim,
                add_cnpt_node=opt.add_cnpt_node)
    else:
        sentence_encoder = BERTSentenceEncoder(
                pretrain_ckpt,
                max_length)
    

    train_data_loader = get_loader(opt.train, sentence_encoder,
            N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, kg_enhance=kg_enrich, use_ernie=opt.use_ernie)
    val_data_loader = get_loader(opt.val, sentence_encoder,
            N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, kg_enhance=kg_enrich, use_ernie=opt.use_ernie)
    test_data_loader = get_loader(opt.test, sentence_encoder,
            N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, kg_enhance=kg_enrich, use_ernie=opt.use_ernie)

   
    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError
    
    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader, cal_meta_rel=cal_meta_rel, keep_grad=keep_grad, lambda_para=opt.lambda_para)

    if model_name == 'proto':
        model = Proto(sentence_encoder, hidden_size=opt.hidden_size, dropout=opt.dropout, feature_size=opt.feature_dim)
    elif model_name == 'my_proto':
        model = MyProto(sentence_encoder, dropout=opt.dropout,  word_embedding_dim=opt.word_embedding_dim, feature_size=opt.feature_dim, hidden_size=opt.hidden_size, add_cnpt_node=opt.add_cnpt_node, kg_enrich=kg_enrich, kg_dim=kg_emb_dim)
    elif model_name == 'proto_meta':
        model = ProtoMeta(sentence_encoder, N, K, dropout=opt.dropout,  word_embedding_dim=opt.word_embedding_dim, feature_size=opt.feature_dim, hidden_size=opt.hidden_size, kg_dim=kg_emb_dim, kg_enrich=kg_enrich, beta=opt.beta, cnpt_att=cnpt_att)
    else:
        raise NotImplementedError
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        framework.train(model, prefix, batch_size, trainN, N, K, Q,
                pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16, pair=opt.pair, 
                train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    acc = framework.eval(model, batch_size, N, K, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair)
    print("RESULT: %.2f" % (acc * 100))

if __name__ == "__main__":
    main()
