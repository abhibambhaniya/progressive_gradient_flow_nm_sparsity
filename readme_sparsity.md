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
- n sparsity: For FF and Conv Layers this corresponding to block wise N:M sparsity in the weight matrix.(int)
- m sparsity: For FF and Conv Layers this corresponding to block wise N:M sparsity in the weight matrix.(int)
- sparsity type : Available paramters are Dense, Structure_NM, SRSTE.
- decay type : For sparse decay, there are 3 types of decay possible, i.e. step, linear and exponential.
- decay coef : For sparse decay of linear and exponential type, this corresponds to the rate of decay.(float: 0-1)
- structure decay flag : This boolean config enables to structure decay of the sparsity pattern. (true,false)
- structure decay type : For structure decay, there are 4 type of decay possible, i.e. 
    sparsify (  Dense -> 7:8    -> 4:8  -> 2:8  -> 1:8 )
    densify ( Dense -> 1:128  -> 1:32 -> 1:16 -> 1:8 )
    fine (  Dense -> 16:128 -> 8:64 -> 4:32 -> 2:16 -> 1:8 )
    custom ( user defined )
- structure decay config : This is user defined config for custom structure decay. ([n1:m1,n2:m2,n3:m3])
- dense epochs : % of epochs that will train the model in dense fashion at the start of the training cycle.(Float: 0-1)
- fine tune epochs : % of epochs that will fine tune the model at the end of the training cycle. (Float 0-1)
- total epochs : Total number of epochs to be trained (Including dense and fine tune).  (int)
- sparsity loc : String input to specify the location of sparsity in the network. 
    FF -> FeedForward
    Q -> Queries
    K -> Keys
    V -> Values
    C -> Convolutions
- n sparsity qkv : For QKV layers, this corresponding to block wise N:M sparsity in the weight matrix.(int)
- m sparsity qkv : For QKV layers, this corresponding to block wise N:M sparsity in the weight matrix.(int)
- sparse dim : This corresponds the direction fo the block for sparsification. By Default we prune along the row. (Row,Col)



## Authors
Abhimanyu Bambhaniya(abambhaniya3@gatech.edu)
Amir Yazdanbhaksh(ayazdan@google.com)



