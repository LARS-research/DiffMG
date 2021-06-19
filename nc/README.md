## Prepare data

```shell
mkdir data
```

We use the preprocessed datasets provided by [GTN](https://github.com/seongjunyun/Graph_Transformer_Networks). Download data from [here](https://drive.google.com/file/d/1qOZ3QjqWMIIvWjzrIdRe3EA4iKzPi6S5/view?usp=sharing) and extract DBLP, ACM and IMDB into `data`.

```shell
python preprocess.py
```

This script will generate `node_types.npy`.

Note that the inputs of our method are only raw information of a heterogeneous network (network topology, node types, edge types, and node attributes if applicable). We do not need to manually design any meta path or meta graph.

 ## Search

To obtain the architectures we use in the paper, run:

```shell
python train_search.py --dataset DBLP --steps 4 --eps 0 --seed 0
python train_search.py --dataset ACM --steps 4 --eps 0.3 --seed 2
python train_search.py --dataset IMDB --steps 4 --eps 0.3 --seed 1
```

Logs are under `log/search/`. The derived architectures are in `arch.py`. To obtain a good architecture, we usually need to run the search algorithm several times with different random seeds.

## Evaluate

Run the following commands to train the derived architectures from scratch:

```shell
python train.py --dataset DBLP --lr 0.005 --wd 0.001 --dropout 0.3
python train.py --dataset ACM --lr 0.01 --wd 0.005 --dropout 0.4 --no_norm --in_nl
python train.py --dataset IMDB --lr 0.01 --wd 0.01 --dropout 0.7
```

Logs are under `log/eval/`. 

For each dataset, we also provide a checkpoint obtained with a good random seed under `checkpoint`. You can evaluate them by:

```shell
python test.py --dataset DBLP
python test.py --dataset ACM --no_norm --in_nl
python test.py --dataset IMDB 
```





