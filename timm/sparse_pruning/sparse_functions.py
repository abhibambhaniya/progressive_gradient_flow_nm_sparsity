""" Sparsity functions in PyTorch

A PyTorch implement of Sparse Linear, Sparse Conv1D and Sparse Conv 2D.

The implementation has linear decay, exponential decay along with structured decay.

Authors: Abhimanyu Bambhaniya
         Amir Yazdanbakhsh
"""

from torch import autograd
import torch.nn.functional as F

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
import sys



# Amir
# 
# SparseSRSTELinear : SR-STE
# Base SR-STE implementation.
class SparseSRSTE(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, sparsity_rate = 0.0, isconv=False, decay = 0.0002):
        ctx.save_for_backward(weight)

        output = weight.clone()
        length = weight.numel()
        
        if sparsity_rate > 0.0:
            num_sparse = int((1.0 - sparsity_rate) * length)
            M = length
            N = num_sparse
        
        group = int(length/M)

        if(isconv == False):
            weight_temp = weight.detach().abs().reshape(group, M)
            index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

            w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
            w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
        else:
            weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
            index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

            w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
            w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.permute(0,2,3,1).shape)
            w_b = w_b.permute(0,3,1,2)

        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b 


    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None, None, None
    
# Rima


class StepDecay(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the default gradient to dense weight in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, sparsity_rate = 0.0, isconv=False):
        ctx.save_for_backward(weight)

        output = weight.clone()
        length = weight.numel()
        
        if sparsity_rate > 0.0:
            num_sparse = int((1.0 - sparsity_rate) * length)
            M = length
            N = num_sparse
        
        group = int(length/M)

        if(isconv == False):
            weight_temp = weight.detach().abs().reshape(group, M)
            index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

            w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
            w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
        else:
            weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
            index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

            w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
            w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.permute(0,2,3,1).shape)
            w_b = w_b.permute(0,3,1,2)
        ctx.mask = w_b
        ctx.decay = 0.0002
        return output*w_b


## For Decay sparse, we are sending grad_output* weight as back prop,
## @amir, is this accurate? 
    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        return grad_output * weight, None, None, None, None

class LinearDecay(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase in a linear fashion but pass the default gradient to dense weight in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, sparsity_rate = 0.0, linear_decay_coef = 0.01 ,current_step_num = 100, isconv=False):
        ctx.save_for_backward(weight)

        output = weight.clone()
        length = weight.numel()
        
        if sparsity_rate > 0.0:
            num_sparse = int((1.0 - sparsity_rate) * length)
            M = length
            N = num_sparse
        
        group = int(length/M)

        if(isconv == False):
            weight_temp = weight.detach().abs().reshape(group, M)
            index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

            w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
            w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
        else:
            weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
            index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

            w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
            w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.permute(0,2,3,1).shape)
            w_b = w_b.permute(0,3,1,2)
        ctx.mask = w_b
        ctx.decay = 0.0002
        mask_decay_value = 1.0
        mask_decay_value = max(
            mask_decay_value -
            (linear_decay_coef*current_step_num),
            0.0)
        if(current_step_num%1000==0):
            print("Linear mask decay value:", mask_decay_value )
        # print(w_b)
        # plt.matshow(w_b.detach().numpy(),cmap=cmap, vmin=-1, vmax=1)
        # plt.matshow(model.hidden1.get_sparse_weights().detach().numpy(),cmap=cmap, vmin=-1, vmax=1)
        # print((1-w_b)*mask_decay_value)
        return output*(w_b + (1-w_b)*mask_decay_value)


## For Decay sparse, we are sending grad_output* weight as back prop,
## @amir, is this accurate? 
    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        return grad_output * weight, None, None, None, None, None, None


class ExponentialDecay(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase in a linear fashion but pass the default gradient to dense weight in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, sparsity_rate = 0.0, exp_decay_coef = 0.01 ,current_step_num = 100, isconv=False):
        ctx.save_for_backward(weight)

        output = weight.clone()
        length = weight.numel()
        
        if sparsity_rate > 0.0:
            num_sparse = int((1.0 - sparsity_rate) * length)
            M = length
            N = num_sparse
        
        group = int(length/M)

        if(isconv == False):
            weight_temp = weight.detach().abs().reshape(group, M)
            index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

            w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
            w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
        else:
            weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
            index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

            w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
            w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.permute(0,2,3,1).shape)
            w_b = w_b.permute(0,3,1,2)
        ctx.mask = w_b
        ctx.decay = 0.0002
        mask_decay_value = math.exp(-1*exp_decay_coef*current_step_num) 
        if(current_step_num%1000==0):
            print("Exponential Mask decay value:", mask_decay_value)
        # print(w_b)
        # plt.matshow(w_b.detach().numpy(),cmap=cmap, vmin=-1, vmax=1)
        # plt.matshow(model.hidden1.get_sparse_weights().detach().numpy(),cmap=cmap, vmin=-1, vmax=1)
        # print((1-w_b)*mask_decay_value)
        return output*(w_b + (1-w_b)*mask_decay_value)


## For Decay sparse, we are sending grad_output* weight as back prop,
## @amir, is this accurate? 
    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        return grad_output * weight, None, None, None, None, None, None


# decay function with linear,exponential,cosine decay. Training step as input, decay value as output.

class SparseLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 sparseConfig = None,**kwargs):
        
        ### Sparsity type
        ##  DENSE = 'DENSE'
        ##  STRUCTURED_NM = 'STRUCTURED_NM'
        ##  UNSTRUCTURED = 'UNSTRUCTURED'
        ##  SRSTE = 'SRSTE'




        self.sparsity_type = sparseConfig.sparsity_type
        ### Decay type
        ## Step :- For all steps enforce the "sparsity_type" for the whole matrix.
        ## Linear :- Linearly decay the mask value from 1 to 0, using the training step number and decay coeffecient.
        ##          Mask decay value = max(0, 1 - linear_decay_coef*(current_step_num-starting_step))
        ##
        ## Exp :- Exponentially decay the mask from 1 to 0, using training step and decay coefficients
        ##          Mask decay value = max(0, e^(-exp_decay_coef*(current_step_num-starting_step))
        ##
        self.decay_type = sparseConfig.decay_type

        ## Structure_decay 
        # Flag to enable uniform structure decay to final N:M.
        ## For example, when target sparsity pattern is 1:16, we divide (n-d-f) 
        # steps to five equal time frame, and the sparsity pattern 
        # of each time frame is 15:16, 8:16, 4:16, 2:16, and 1:16, respectively.
        self.structure_decay_flag = sparseConfig.structure_decay_flag

        ## Sparsity % for uniform sparsity
        self.N = sparseConfig.n_sparsity
        self.M = sparseConfig.m_sparsity
        self.sparsity_rate = sparseConfig.prune_rate

        print("Enabling sparsity type:",self.sparsity_type," , decay_type: ",self.decay_type,", Structure decay flag:",self.structure_decay_flag,", N:M=",self.N,self.M)
        ## Decay config params
        self.decay_coef = sparseConfig.decay_coef
        self.current_step_num = 0
        self.current_epoch = 0

        ## TBD, update these from real parameters in training run.
        self.dense_epochs = sparseConfig.dense_epochs
        self.fine_tune_epochs = sparseConfig.fine_tune_epochs
        self.total_epochs = sparseConfig.total_epochs


        if(self.structure_decay_flag == True):
            num_sparse_frames = int(math.log2(self.M/self.N))
            sparse_frame_size = (self.total_epochs - self.dense_epochs - self.fine_tune_epochs) / num_sparse_frames
            self.structure_decay_config = dict()
            for i in range(num_sparse_frames+1):
                self.structure_decay_config[int(self.dense_epochs) + int(i*sparse_frame_size)]  = str((self.M-1) if (i == 0) else int(self.M/math.pow(2,i))) + ":" + str(self.M)
            print(self.structure_decay_config)
        else:
            self.structure_decay_config = None

        ## To be implemented Custom structure decay schedule.
        ### Structure decay config
        ## Ex: - The (n-d-s) steps are divided into user config for [20=3:4,50=4:8, 70=1:4], For this config 
        # from 20% of steps, 3:4 is followed, next, from 50% of steps, 4:8 is followed and 
        # finally from 70% onwards, 1:8 is followed.
        # if(sparseConfig.structure_decay_config is not None):
        #     self.structure_decay_config = sparseConfig.structure_decay_config
        # elif(self.structure_decay_flag == True):
        #     num_sparse_frames = int(math.log2(self.M/self.N))
        #     sparse_frame_size = (self.total_steps - self.dense_steps - self.fine_tune_steps) / num_sparse_frames
        #     self.structure_decay_config = dict()
        #     for i in range(num_sparse_frames):
        #         self.structure_decay_config[int(self.dense_steps) + int(i*sparse_frame_size)]  = str((self.M-1) if (i == 0) else int(self.M/math.pow(2,i))) + ":" + str(self.M)
        #     # print(self.structure_decay_config)
        # else:
        


        super(SparseLinear, self).__init__(in_features, out_features, bias = True)


    def get_sparse_weights(self):
        if(self.current_epoch < self.dense_epochs or self.current_step_num<0):     ## Return dense weights
            return self.weight

        if(self.sparsity_type.lower() == "srste"):
            return SparseSRSTE.apply(self.weight, self.N, self.M, self.sparsity_rate)
        else:
            if(self.structure_decay_config is not None):
                if(self.current_epoch in self.structure_decay_config ):
                    self.N =  int(self.structure_decay_config[self.current_epoch].split(':')[0])
                    self.M =  int(self.structure_decay_config[self.current_epoch].split(':')[1])
#                     print("Updating the weights to ",self.N,":",self.M)

            ## When doing fine tuning, the mask is binary (0,1)
            if(self.current_epoch > (self.total_epochs-self.fine_tune_epochs)):
                self.decay_type = "step"


            if(self.decay_type.lower() == "step"):
                return StepDecay.apply(self.weight, self.N, self.M, self.sparsity_rate)
            elif(self.decay_type.lower() == "linear"):
                return LinearDecay.apply(self.weight, self.N, self.M, self.sparsity_rate, self.decay_coef,(self.current_step_num))
            elif(self.decay_type.lower() == "exp"):
                return ExponentialDecay.apply(self.weight, self.N, self.M, self.sparsity_rate, self.decay_coef,(self.current_step_num))
            else:
                print("decay type unidentified. Use on of the following: step,linear,exp.")
                sys.exit(1)



    def forward(self, x, current_step_num = 0,current_epoch=0):
        self.current_step_num = current_step_num
        self.current_epoch = current_epoch
#         print("current step num:",current_step_num) 
        w = self.get_sparse_weights()
        x = F.linear(x, w)
        return x





# class SparseConv1D(nn.Conv1D):

#     def __init__(self, in_features: int, out_features: int, bias: bool = True,
#                  config = None,**kwargs):
        
#         ### Sparsity type
#         ## SRSTE :- baseline implementation of SR-STE
#         ## 
        
#         self.sparsity_type = config.sparsity_type
#         ### Decay type
#         ## Step :- For all steps enforce the "sparsity_type" for the whole matrix.
#         ## Linear :- Linearly decay the mask value from 1 to 0, using the training step number and decay coeffecient.
#         ##          Mask decay value = max(0, 1 - linear_decay_coef*(current_step_num-starting_step))
#         ##
#         ## Exp :- Exponentially decay the mask from 1 to 0, using training step and decay coefficients
#         ##          Mask decay value = max(0, e^(-exp_decay_coef*(current_step_num-starting_step))
#         ##
#         self.decay_type = config.decay_type

#         ## Structure_decay 
#         # Flag to enable uniform structure decay to final N:M.
#         ## For example, when target sparsity pattern is 1:16, we divide (n-d-s) 
#         # steps to five equal time frame, and the sparsity pattern 
#         # of each time frame is 15:16, 8:16, 4:16, 2:16, and 1:16, respectively.
#         self.structure_decay_flag = config.structure_decay_flag

#         ## Sparsity % for uniform sparsity
#         self.N = config.n_sparsity
#         self.M = config.m_sparsity
#         self.sparsity_rate = config.unstructured_sparsity_rate

#         ## Decay config params
#         self.linear_decay_coef = config.linear_decay_coef
#         self.exp_decay_coef = config.exp_decay_coef
#         self.current_step_num = 0

#         ## TBD, update these from real parameters in training run.
#         self.dense_steps = 10
#         self.fine_tune_steps = config.fine_tune_steps
#         self.total_steps = 120


#         ## To be implemented Custom structure decay schedule.
#         ### Structure decay config
#         ## Ex: - The (n-d-s) steps are divided into user config for [20=3:4,50=4:8, 70=1:4], For this config 
#         # from 20% of steps, 3:4 is followed, next, from 50% of steps, 4:8 is followed and 
#         # finally from 70% onwards, 1:8 is followed.
#         if(config.structure_decay_config is not None):
#             self.structure_decay_config = config.structure_decay_config
#         else:
#             self.structure_decay_config = None


#         super(SparseConv1D, self).__init__(out_features, in_features)

#     def get_sparse_weights(self):
#         if(self.sparsity_type.lower() == "srste"):
#             return SparseSRSTE.apply(self.weight, self.N, self.M, self.sparsity_rate)
#         else:
#             if(self.structure_decay_config is not None):
#                 if(self.current_step_num in self.structure_decay_config):
#                     self.N =  int(self.structure_decay_config[self.current_step_num].split(':')[0])
#                     self.M =  int(self.structure_decay_config[self.current_step_num].split(':')[1])
#                     print("Updating the weights to ",self.N,":",self.M)

#             ## When doing fine tuning, the mask is binary (0,1)
#             if(self.current_step_num > (self.total_steps-self.fine_tune_steps)):
#                 self.decay_type = "Step"


#             if(self.decay_type.lower() == "step"):
#                 return StepDecay.apply(self.weight, self.N, self.M, self.sparsity_rate)
#             elif(self.decay_type.lower() == "linear"):
#                 return LinearDecay.apply(self.weight, self.N, self.M, self.sparsity_rate, self.linear_decay_coef,(self.current_step_num-self.dense_steps))
#             elif(self.decay_type.lower() == "exp"):
#                 return ExponentialDecay.apply(self.weight, self.N, self.M, self.sparsity_rate, self.exp_decay_coef,(self.current_step_num-self.dense_steps))
#             else:
#                 print("decay type unidentified. Use on of the following: step,linear,exp.")
#                 sys.exit(1)

#     def update_global_step_num(self, start_step: int = 0, max_steps: int = 0):
#         print("Updating dense steps to ", self.dense_steps, " and total_steps to ",self.total_steps)
#         self.dense_steps = start_step
#         self.total_steps = max_steps
#         if(self.structure_decay_config is None and self.structure_decay_flag == True):
#             num_sparse_frames = int(math.log2(self.M/self.N))
#             sparse_frame_size = (self.total_steps - self.dense_steps - self.fine_tune_steps) / num_sparse_frames
#             self.structure_decay_config = dict()
#             for i in range(num_sparse_frames+1):
#                 self.structure_decay_config[int(self.dense_steps) + int(i*sparse_frame_size)]  = str((self.M-1) if (i == 0) else int(self.M/math.pow(2,i))) + ":" + str(self.M)
#             print(self.structure_decay_config)
#         else:
#             self.structure_decay_config = None

#     def forward(self, x, current_step_num = 0):
#         self.current_step_num = current_step_num
#         w = self.get_sparse_weights()
#         size_out = x.size()[:-1] + (self.nf,)
#         x = torch.addmm(self.bias, x.view(-1, x.size(-1)), w)
#         x = x.view(size_out)
#         return x



class SparseConv2D(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 sparseConfig = None,**kwargs):
        
        ### Sparsity type
        ##  DENSE = 'DENSE'
        ##  STRUCTURED_NM = 'STRUCTURED_NM'
        ##  UNSTRUCTURED = 'UNSTRUCTURED'
        ##  SRSTE = 'SRSTE'




        self.sparsity_type = sparseConfig.sparsity_type
        ### Decay type
        ## Step :- For all steps enforce the "sparsity_type" for the whole matrix.
        ## Linear :- Linearly decay the mask value from 1 to 0, using the training step number and decay coeffecient.
        ##          Mask decay value = max(0, 1 - linear_decay_coef*(current_step_num-starting_step))
        ##
        ## Exp :- Exponentially decay the mask from 1 to 0, using training step and decay coefficients
        ##          Mask decay value = max(0, e^(-exp_decay_coef*(current_step_num-starting_step))
        ##
        self.decay_type = sparseConfig.decay_type

        ## Structure_decay 
        # Flag to enable uniform structure decay to final N:M.
        ## For example, when target sparsity pattern is 1:16, we divide (n-d-f) 
        # steps to five equal time frame, and the sparsity pattern 
        # of each time frame is 15:16, 8:16, 4:16, 2:16, and 1:16, respectively.
        self.structure_decay_flag = sparseConfig.structure_decay_flag

        ## Sparsity % for uniform sparsity
        self.N = sparseConfig.n_sparsity
        self.M = sparseConfig.m_sparsity
        self.sparsity_rate = sparseConfig.prune_rate

        print("Enabling sparsity type:",self.sparsity_type," , decay_type: ",self.decay_type,", Structure decay flag:",self.structure_decay_flag,", N:M=",self.N,self.M)
        ## Decay config params
        self.decay_coef = sparseConfig.decay_coef
        self.current_step_num = 0
        self.current_epoch = 0

        ## TBD, update these from real parameters in training run.
        self.dense_epochs = sparseConfig.dense_epochs
        self.fine_tune_epochs = sparseConfig.fine_tune_epochs
        self.total_epochs = sparseConfig.total_epochs


        if(self.structure_decay_flag == True):
            num_sparse_frames = int(math.log2(self.M/self.N))
            sparse_frame_size = (self.total_epochs - self.dense_epochs - self.fine_tune_epochs) / num_sparse_frames
            self.structure_decay_config = dict()
            for i in range(num_sparse_frames+1):
                self.structure_decay_config[int(self.dense_epochs) + int(i*sparse_frame_size)]  = str((self.M-1) if (i == 0) else int(self.M/math.pow(2,i))) + ":" + str(self.M)
            print(self.structure_decay_config)
        else:
            self.structure_decay_config = None


        super(SparseConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)

    def get_sparse_weights(self):
        if(self.current_epoch < self.dense_epochs or self.current_step_num<0):     ## Return dense weights
            return self.weight

        if(self.sparsity_type.lower() == "srste"):
            return SparseSRSTE.apply(self.weight, self.N, self.M, self.sparsity_rate, True)
        else:
            if(self.structure_decay_config is not None):
                if(self.current_epoch in self.structure_decay_config ):
                    self.N =  int(self.structure_decay_config[self.current_epoch].split(':')[0])
                    self.M =  int(self.structure_decay_config[self.current_epoch].split(':')[1])
#                     print("Updating the weights to ",self.N,":",self.M)

            ## When doing fine tuning, the mask is binary (0,1)
            if(self.current_epoch > (self.total_epochs-self.fine_tune_epochs)):
                self.decay_type = "step"


            if(self.decay_type.lower() == "step"):
                return StepDecay.apply(self.weight, self.N, self.M, self.sparsity_rate, True)
            elif(self.decay_type.lower() == "linear"):
                return LinearDecay.apply(self.weight, self.N, self.M, self.sparsity_rate, self.decay_coef,(self.current_step_num), True)
            elif(self.decay_type.lower() == "exp"):
                return ExponentialDecay.apply(self.weight, self.N, self.M, self.sparsity_rate, self.decay_coef,(self.current_step_num), True)
            else:
                print("decay type unidentified. Use on of the following: step,linear,exp.")
                sys.exit(1)



    def forward(self, x, current_step_num = 0,current_epoch=0):
        self.current_step_num = current_step_num
        self.current_epoch = current_epoch
        w = self.get_sparse_weights()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x 
