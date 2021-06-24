# DiffMG
This repository contains the code for our KDD 2021 Research Track paper: *DiffMG: Differentiable Meta Graph Search for Heterogeneous Graph Neural Networks*.
https://arxiv.org/abs/2010.03250

<img width="998" alt="Screenshot 2021-06-22 at 3 08 06 PM" src="https://user-images.githubusercontent.com/22978940/122879585-a77ed280-d36b-11eb-9b49-6e8fed99faba.png">

## Environment

```
python=3.8
pytorch==1.6.0 with CUDA support (by default our model is trained on GPU)
numpy==1.19.1
scipy==1.5.2
pandas==1.1.1
scikit-learn==0.23.2
```

## How to run

For the node classification task, please see README under `nc`, and for the recommendation task, please see README under `lp`.

## Citation

If you find our work helpful in your own research, please consider citing our paper:


```
@inproceedings{diffmg,
  title={DiffMG: Differentiable Meta Graph Search for Heterogeneous Graph Neural Networks},
  author={Ding, Yuhui and Yao, Quanming and Zhao, Huan and Zhang, Tong},
  booktitle={Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2021}
}
```
