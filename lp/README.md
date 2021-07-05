## File Description

`preprocessed/`: Preprocessed data.

`preprocess.py`: Preprocessing script.

`gen_neg.py`: Preprocessing script.

`model_search.py`: The DAG search space (i.e., the supernet).

`train_search.py`: The script to perform search. Please run `python train_search.py --help` to see command-line arguments and what they mean.

`arch.py`: The architectures we use for evaluation are recorded here.

`model.py`: The wrap of an arbitrary architecture derived from the search space.

`train.py`: This script imports `arch.py` and trains the discovered architectures from scratch. Please run `python train.py --help` to see command-line arguments and what they mean.

## Preparing Data

**We already provide the necessary preprocessed data under `preprocessed/`.** 

`adjs_offset.pkl` is a dict whose values are scipy sparse matrices. The keys only serve as identifiers and can be arbitrary. Each matrix has a shape of (num_of_nodes, num_of_nodes) and is formed by edges of a certain edge type. To save space, for two edge types which are right opposite (e.g., user-movie and movie-user), we only store one of them.

`node_types.npy` is a numpy array with a shape of (num_of_nodes,). Each value in the array is an integer and lies in [0, num_of_node_types).

`pos_pairs_offset.npz` contains source-target pairs with label "1". pos_pairs_offset["train"] is a numpy array with a shape of (num_of_training_pairs, 2) and each row has the form [source_id, target_id]. pos_pairs_offset["val"] and pos_pairs_offset["test"] are for validation and testing respectively.

`neg_pairs_offset.npz` contains source-target pairs with label "0", with the same format as `pos_pairs_offset.npz`.


The following steps show how we generate them from raw data.

```shell
mkdir data
```

Copy datasets from https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding into `data/` (rename `Douban Movie` as `Douban_Movie`)

```shell
python preprocess.py Yelp 1
python gen_neg.py Yelp 2
python preprocess.py Amazon 3
python gen_neg.py Amazon 4
python preprocess.py Douban_Movie 5
python gen_neg.py Douban_Movie 6
```
1-6 are random seeds for reproducible dataset splitting. The most time-consuming part in `preprocess.py` is a nested for loop to count unconnected source-target pairs, so it takes several hours to finish.

Note that the inputs of our method are only raw information of a heterogeneous network (network topology, node types, edge types, and node attributes if applicable). We do not need to manually design any meta path or meta graph.

## Search

To obtain the architectures we use in the paper, run:

```shell
python train_search.py --dataset Amazon --steps_s 4 --steps_t 4 --lr 0.01 --eps 0.5 --seed 0
python train_search.py --dataset Yelp --steps_s 4 --steps_t 4 --seed 1
python train_search.py --dataset Douban_Movie --steps_s 4 --steps_t 4 --seed 1
```

Logs are automatically generated under `log/search/` with the following format: 

```
epoch number; training error; validation error; architecture for source node type; architecture for target node type
```

To obtain a good architecture, we usually need to run the search algorithm several times with different random seeds.

## Architecture Interpretation

Similar to the node classification task. Please refer to README therein.

## Evaluation

Run the following commands to train the derived architectures from scratch:

```shell
python train.py --dataset Amazon --lr 0.01 --wd 0.001 --dropout 0.6
python train.py --dataset Yelp --lr 0.01 --wd 0.0005 --dropout 0.6
python train.py --dataset Douban_Movie --lr 0.01 --wd 0.0003 --dropout 0.2
```

Logs are automatically generated under `log/eval/`. The checkpoint that achieves the highest validation AUC is used for testing.
