# Structured sparsification of Models: Pytorch implementaion
- [Introduction](#introduction)
- [Parameters](#parameters)
- [Authors](#authors)

## Introduction

The repo has been forked from huggingface's pytorch image models repo(https://github.com/rwightman/pytorch-image-models). 
We have implemented various different techniques to structured sparsification to the repo.

Currently we have following models with sparsity
- ViT ( Sparsity in FF layers and QKV) [Verified]
- Resnet (Sparsity in Conv layers ) [Verified]
- Swin (Sparsity in FF and QKV ) [In development]

There are various different types of structured sparsification training from scratch recipe available.
To enable sparsity add appropriate parameters to the config file.

## Parameters
* N Sparsity: For FF and Conv Layers this corresponding to block wise N:M sparsity in the weight matrix.(int)
* M Sparsity: For FF and Conv Layers this corresponding to block wise N:M sparsity in the weight matrix.(int)
* Sparsity Type : Available paramters are Dense, Structure_NM, SRSTE.
* Decay Type : For sparse decay, there are 3 types of decay possible, i.e. step, linear and exponential.
* Decay Coef : For sparse decay of linear and exponential type, this corresponds to the rate of decay.(float: 0-1)
* Structure Decay Flag : This boolean config enables to structure decay of the sparsity pattern. (true,false)
* Structure Decay Type : For structure decay, there are 4 type of decay possible, i.e. 
    * sparsify (  Dense -> 7:8    -> 4:8  -> 2:8  -> 1:8 )
    * densify ( Dense -> 1:128  -> 1:32 -> 1:16 -> 1:8 )
    * fine (  Dense -> 16:128 -> 8:64 -> 4:32 -> 2:16 -> 1:8 )
    * custom ( user defined )
* Structure Decay Config : This is user defined config for custom structure decay. ([n1:m1,n2:m2,n3:m3])
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
Abhimanyu Bambhaniya(abambhaniya3@gatech.edu)

Amir Yazdanbhaksh(ayazdan@google.com)



