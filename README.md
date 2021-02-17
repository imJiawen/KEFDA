# KEFDA
Code and KGs for paper "Knowledge-Enhanced Domain Adaptation in Few-Shot Relation Classification" submitted in KDD'2021.

## Data Preparation

**FewRel** and **FewRel 2.0 Datasets** are availble at [here](https://github.com/thunlp/FewRel). 
Please download the *data* folder and place it in the home directory.

### Use a pre-processed knowledge graph

The pre-processed **Knowledge Graph** can be downloaded directly from [Google Drive](https://drive.google.com/drive/folders/1t5DUvE3tedLyWJLlT1H1iYIvNY2u3xxm?usp=sharing). \
Please place the *ontology* folder at ```./fewshot_re_kit/kg_toolkit/```.

### Use your own knowledge graph

<!--you can also choose to use your own knowledge graph by pre-processing it into the following format. -->

Guidance on how to pre-process your own knowledge graph will be released soon....  :construction:


## How to Run

### Training a Model

To run our model, use command

> bash train.sh

This will start the training and evaluating process of KEDAF in a 5-way 1-shot setting.

You can train the model in different configurations by setting different args in ```train.sh```.

- ```train / val / test```: Specify the training / validation / test set. 
- ```N```: N in N-way K-shot.
- ```K```: N in N-way K-shot.
- ```Q```: Sample Q query instances for each relation.
- ```kg_encoder```: Which knowledge graph encoder is used to encode the graph (only need to be set for the provided pre-trained graph embedding).
- ```cnpt_att```: Whether to aggregate relation meta using the attention mechanism.
- ```lambda_para```: Trade-off the importance of the knowledge enhanced prototype network and the relation-meta learning network, and the value should be picked between [0,1]. The higher the value, the more weight is given to the prototype network.

### Inference

Run the following command to evaluate the existing checkpoint:

> python train.py --only_test --load_ckpt {CHECKPOINT_PATH} {OTHER_ARGS}

