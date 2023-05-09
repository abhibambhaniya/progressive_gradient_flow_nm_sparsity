""" Summary utilities

Hacked together by / Copyright 2020 Ross Wightman
"""
import csv
import os
from collections import OrderedDict
import torch
try: 
    import wandb
except ImportError:
    pass


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def update_summary(
        epoch,
        train_metrics,
        eval_metrics,
        filename,
        lr=None,
        write_header=False,
        log_wandb=False,
):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if lr is not None:
        rowd['lr'] = lr
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)

## Abhi
def get_sparse_weights(weight,N=2,M=4):
    length = weight.numel()
    
    group = int(length/M)

    weight_temp = weight.detach().abs().reshape(group, M)
    index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

    w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
    w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)

    return w_b*weight

def Update_model_stats(
        step,
        model,
        args,
        filename,

):
    rowd = OrderedDict(step_num=step)
    for weight in model.state_dict():
        if('fc' in weight and 'weight' in weight):
            weight_matrix = model.state_dict()[weight] 
            rowd.update({f'mean_' + str(weight) : torch.mean(weight_matrix).cpu().numpy()})
            rowd.update({f'std_' + str(weight) :  torch.std(weight_matrix).cpu().numpy()})
            sparse_weight_matrix = get_sparse_weights(weight_matrix, args.n_sparsity, args.m_sparsity)
            rowd.update({f'mean_sparse_' + str(weight) : torch.mean(sparse_weight_matrix).cpu().numpy()})
            rowd.update({f'std_sparse_' + str(weight) :  torch.std(sparse_weight_matrix).cpu().numpy()})

    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        # Get the size of the file
        file_size = os.path.getsize(filename)
    
        # Check if the file is empty
        if file_size == 0:
            dw.writeheader()
        dw.writerow(rowd)


## Ihba