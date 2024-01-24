# Structured sparsification of Models: Pytorch implementaion
- [Introduction](#introduction)
- [Examples](#examples)
- [Parameters](#parameters)
- [Authors](#authors)

## Introduction

The repo has been forked from huggingface's pytorch image models [**repo**](https://github.com/rwightman/pytorch-image-models). 
We have implemented various different techniques to structured sparsification to the repo.

Currently we have following models with sparsity
- ViT ( Sparsity in FF layers and QKV) [Verified]
- Swin (Sparsity in FF and QKV ) [Verfied]

There are various different types of structured sparsification training from scratch recipe available.

## Examples
To Install the dependencies.

$ pip install -e .

We have added 1 configuration for running the model training, for ViT .

For running on a single system

python3 ./train.py --config vit_base_training.yaml 

For running on distributed system with multiple GPUs

./distributed_train.sh --config vit_base_training.yaml

We build a similar yaml file for swin_v2, use the base hyper parameters from original swin repo.
https://github.com/microsoft/Swin-Transformer/blob/d19503d7fbed704792a5e5a3a5ee36f9357d26c1/config.py



Various paramters to enable sparsity are added in the config file. We can tweek them as required.

## Parameters
* N Sparsity: For FF and Conv Layers this corresponding to block wise N:M sparsity in the weight matrix.(int)
* M Sparsity: For FF and Conv Layers this corresponding to block wise N:M sparsity in the weight matrix.(int)

* Sparsity Type : Available paramters are Dense, Structure_NM, SRSTE.

* Decay Type : This represents different types of MDGF. For sparse decay, there are 3 types of decay possible, i.e. step, linear and exponential.
* Decay Coef : For sparse decay of linear and exponential type, this corresponds to the rate of decay.(float: 0-1)

* Structure Decay Flag : This represents different types of SDGF.
                         This boolean flag enables to structure decay of the sparsity pattern. (true,false)
* Structure Decay Type : For structure decay, there are 2 type of decay possible, i.e. 
    * "sparsify", SdGF-Stepwise (  Dense -> 7:8    -> 4:8  -> 2:8  -> 1:8 )
    * "fine", SdGf-Geometric (  Dense -> 8:64 -> 4:32 -> 2:16 -> 1:8 )


* Dense Epochs : % of epochs that will train the model in dense fashion at the start of the training cycle.(Float: 0-1)
* Fine Tune Epochs : % of epochs that will fine tune the model at the end of the training cycle. (Float 0-1)
* Total Epochs : Total number of epochs to be trained (Including dense and fine tune).  (int)
* Sparsity Loc : String input to specify the location of sparsity in the network. 
    * FF -> FeedForward
    * Q -> Queries
    * K -> Keys
    * V -> Values
    * C -> Convolutions
* N Sparsity qkv : For QKV layers, this corresponding to block wise N:M sparsity in the weight matrix.(int)
* M Sparsity qkv : For QKV layers, this corresponding to block wise N:M sparsity in the weight matrix.(int)
* Sparse Dim : This corresponds the direction fo the block for sparsification. By Default we prune along the row. (Row,Col)



## Authors
Anonimized for ICML 2024 Submission

