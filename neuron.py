import torch
from torch import nn
from function import ActFun

class BaseNode(nn.Module):
    def __init__(self) -> None:
        super(BaseNode,self).__init__()

    def reset(self) -> None:
        if hasattr(self,'mem'):
            self.mem=0
        if hasattr(self,'spike_pot'):
            self.spike_pot=[]
        if hasattr(self,'cache'):
            self.cache=0

# class LIFNode(BaseNode):
#     def __init__(self,v_threshold:float=0.5,v_reset:float=0.0,tau:float=5,surrogate_type:str='sigmoid',surrogate_param:float=2.0) -> None:
#         super(LIFNode,self).__init__()
#         self.v_reset=v_reset
#         self.v_threshold=v_threshold
#         self.tau=tau
#         self.activate_function=ActFun.apply
#         self.surrogate_type=surrogate_type
#         self.surrogate_param=surrogate_param
#         self.mem=0
#         self.spike_pot=[]

#     def forward(self,X:torch.Tensor) -> torch.Tensor:
#         self.reset()
#         T=X.shape[1]
#         for t in range(T):
#             self.mem=self.mem/self.tau+X[:,t,...]
#             spike=self.activate_function(self.mem-self.v_threshold,self.surrogate_type,self.surrogate_param)
#             if self.v_reset is None:
#                 self.mem=self.mem-spike*self.v_threshold
#             else:
#                 self.mem=(1-spike)*self.mem+self.v_reset*spike
#             self.spike_pot.append(spike)
#         return torch.stack(self.spike_pot,dim=1)

class LIFNode(BaseNode):
    def __init__(self,v_threshold:float=0.5,v_reset:float=0.0,tau:float=5,alpha:float|nn.Parameter=0.1,beta:float|nn.Parameter=0.5,
                 surrogate_type:str='sigmoid',surrogate_param:float=2.0) -> None:
        super(LIFNode,self).__init__()
        self.v_reset=v_reset
        self.v_threshold=v_threshold
        self.alpha=alpha
        self.beta=beta
        self.tau=tau
        self.cache=0
        self.activate_function=ActFun.apply
        self.surrogate_type=surrogate_type
        self.surrogate_param=surrogate_param
        self.mem=0
        self.spike_pot=[]

    def forward(self,X:torch.Tensor) -> torch.Tensor:
        self.reset()
        T=X.shape[1]
        for t in range(T):
            self.mem=self.mem/self.tau+(1+self.get_param(self.alpha)*self.cache)*X[:,t,...]
            spike=self.activate_function(self.mem-self.v_threshold,self.surrogate_type,self.surrogate_param)
            if self.v_reset is None:
                self.mem=self.mem-spike*self.v_threshold
            else:
                self.mem=(1-spike)*self.mem+self.v_reset*spike
            self.spike_pot.append(spike)
            # self.cache=(1-self.beta)*self.cache+self.beta*(1-spike)
            self.cache=self.get_param(self.beta)*self.cache+(1-spike)
        return torch.stack(self.spike_pot,dim=1)
    
    def get_param(self,param:float|nn.Parameter) -> float|nn.Parameter:
        return param.sigmoid() if isinstance(param,nn.Parameter) else param
