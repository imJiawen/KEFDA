# KEFDA
Code and KGs for paper "Knowledge-Enhanced Domain Adaptation in Few-Shot Relation Classification" submitted in KDD'2021.

## Data Preparation

**FewRel** and **FewRel 2.0 Datasets** are availble at [here](https://github.com/thunlp/FewRel). 
Please download the *data* folder and place it in the home directory.

### Use a pre-processed knowledge graph

The pre-processed **Knowledge Graph** can be downloaded directly from [Google Drive](https://drive.google.com/drive/folders/1t5DUvE3tedLyWJLlT1H1iYIvNY2u3xxm?usp=sharing). \
Please place the *ontology* folder at ```./fewshot_re_kit/kg_toolkit/```.

### Use your own knowledge graph

You can also choose to use your own knowledge graph by pre-processing it into a specific format.

1. Convert your knowledge graph into a low-dimensional representation, and we use [OpenKE Toolkit](https://github.com/thunlp/OpenKE) for this process.
Please place the learned embedding in the ```./fewshot_re_kit/kg_toolkil/ontology/kg_embedding``` folder.
2. Please set the ```--kg_encoder``` argument in ```train.sh``` to the kg embedding file name, i.e. ```--kg_encoder my_transe.embedding```.
3. Please place the following files in the ```./fewshot_re_kit/kg_toolkil/ontology``` folder:
    - entity2id.txt
    ```
    # total entities
    ENTITY_1  id_1
    ENTITY_2  id_2
    ...
    ```
    - ent2cnpt.txt
    ```
    ENTITY  CNPT1   CNPT2   ...
    ```
    - triple2id.txt
    ```
    # total triples
    head_entity_id rel_id tail_entity_id
    ...
    ```
    - concept2id.txt
    ```
    # total concepts
    CNPT_1  id_1
    ...
    ```
    - cnpt_id2def_fea.txt (The description features of concepts can be produced by [BERT](https://github.com/huggingface/transformers)).
    ```
    CNPT_id CNPT_DESC_FEATURES
    ```
    - desc_feature/entity2id.txt
    ```
    ENTITY id
    ```
    - desc_feature/id2fea.txt
    ```
    ENT_id ENT_DESC_FEATURES
    ```
    

<!--Guidance on how to pre-process your own knowledge graph will be released soon....  :construction: -->


## How to Run

### Training a Model

To run our model, use command

```
bash train.sh
```

This will start the training and evaluating process of KEDAF in a 5-way 1-shot setting.

You can train the model in different configurations by setting different args in `train.sh`.

- ```train / val / test```: Specify the training / validation / test set. 
- ```N```: N in N-way K-shot.
- ```K```: N in N-way K-shot.
- ```kg_encoder```: Which knowledge graph encoder is used to encode the graph (for the provided pre-trained graph embedding, choose between transe/distmult/analogy/rotate; otherwise, set it to the name of the customized embedding file).
- ```cnpt_att```: Whether to aggregate relation meta using the attention mechanism.
- ```lambda_para```: Trade-off the importance of the knowledge enhanced prototype network and the relation-meta learning network, and the value should be picked between [0,1]. The higher the value, the more weight is given to the prototype network.

More arguments please see ```train.py``` for detail.

### Inference

Run the following command to evaluate the existing checkpoint:

```
python train.py --only_test --load_ckpt {CHECKPOINT_PATH} {OTHER_ARGS}
```

