import math
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self,m,a,b,g,scale=None):
        super().__init__(); self.w=nn.Parameter(torch.randn(m,a,b,generator=g)*(1/math.sqrt(a) if scale is None else scale)); self.b=nn.Parameter(torch.zeros(m,1,b))
    def forward(self,x):return torch.bmm(x,self.w)+self.b

class Network(nn.Module):
    def __init__(self,m,family,width,seed):
        super().__init__();self.family=family;self.width=width;g=torch.Generator().manual_seed(seed)
        if family=='DNN':self.f1=Linear(m,3,width,g);self.f2=Linear(m,width,width,g)
        else:self.gates=Linear(m,3+width,4*width,g)
        self.out=Linear(m,width,1,g,.02)
    def acceleration(self,x,mem=None):
        if self.family=='DNN':return self.out(torch.tanh(self.f2(torch.tanh(self.f1(x)))))[:,:,0],None
        if mem is None:mem=(x.new_zeros((*x.shape[:2],self.width)),x.new_zeros((*x.shape[:2],self.width)))
        hh,cc=mem;i,f,o,j=self.gates(torch.cat([x,hh],-1)).chunk(4,-1)
        cc=torch.sigmoid(f)*cc+torch.sigmoid(i)*torch.tanh(j);hh=torch.sigmoid(o)*torch.tanh(cc)
        return self.out(hh)[:,:,0],(hh,cc)

def advance(x,acc,u0,u1,dt,mu,sc):
    z=x*sc+mu;s,v,d=z.unbind(-1);vn=v+dt*acc;sn=s+.5*dt*(u0+u1-v-vn)
    return (torch.stack([sn,vn,u1-vn],-1)-mu)/sc
