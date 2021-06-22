## Prepare data

**We already provide the necessary preprocessed data under the `preprocessed` folder, so you can search and evaluate directly.** 

The following steps show how we generate them from raw data.

```shell
mkdir data
```

Copy datasets from https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding into `data` (rename `Douban Movie` as `Douban_Movie`)

```shell
python preprocess.py Yelp 1
python gen_neg.py Yelp 2
python preprocess.py Amazon 3
python gen_neg.py Amazon 4
python preprocess.py Douban_Movie 5
python gen_neg.py Douban_Movie 6
```

The main purpose of `preprocess.py` is to generate adjacency matrices formed by different edge types (`adjs_offset.pkl`), and `gen_neg.py` splits negative pairs (i.e., with label "0"). 1-6 are random seeds for reproducible dataset splitting. The most time-consuming part in `preprocess.py` is a nested for loop to count unconnected source-target pairs (which may serve as negative pairs), so it takes several hours to finish.

Note that the inputs of our method are only raw information of a heterogeneous network (network topology, node types, edge types, and node attributes if applicable). We do not need to manually design any meta path or meta graph.

## Search

To obtain the architectures we use in the paper, run:

```shell
python train_search.py --dataset Amazon --steps_s 4 --steps_t 4 --lr 0.01 --eps 0.5 --seed 0
python train_search.py --dataset Yelp --steps_s 4 --steps_t 4 --seed 1
python train_search.py --dataset Douban_Movie --steps_s 4 --steps_t 4 --seed 1
```

Logs are under `log/search/`. The derived architectures are in `arch.py`. To obtain a good architecture, we usually need to run the search algorithm several times with different random seeds.

## Evaluate

Run the following commands to train the derived architectures from scratch:

```shell
python train.py --dataset Amazon --lr 0.01 --wd 0.001 --dropout 0.6
python train.py --dataset Yelp --lr 0.01 --wd 0.0005 --dropout 0.6
python train.py --dataset Douban_Movie --lr 0.01 --wd 0.0003 --dropout 0.2
```

Logs are under `log/eval/`.
