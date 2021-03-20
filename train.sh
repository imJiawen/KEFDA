python train.py \
     --train_iter 20000 \
     --trainN 5 --N 5 --K 5 --Q 1 \
     --val val_pubmed --test val_pubmed \
     --model proto_meta --encoder bert --hidden_size 768 --word_embedding_dim 768 --feature_dim 768  \
     --optim adam --lr 1e-4 --batch_size 1 --val_step 1000 --gpu 0 \
     --kg_dict ./fewshot_re_kit/kg_toolkit/ontology --add_cnpt_node \
     --kg_encoder "distmult" --beta 1 \
     --lambda_para 0.95 --cnpt_att \
