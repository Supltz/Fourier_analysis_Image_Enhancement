from typing_extensions import Required
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
class Net(nn.Module):
    def __init__(self,N):
        super().__init__()
        self.real = nn.Linear(N,N,bias=False)
        self.imag = nn.Linear(N, N, bias=False)

    def forward(self,x):
      def matmul_complex(t1,t2):
          return torch.view_as_complex(torch.stack((t1.real @ t2.real - t1.imag @ t2.imag, t1.real @ t2.imag + t1.imag @ t2.real),dim=1))
      fft_in_rows = torch.zeros([len(x),len(x)]).to(torch.cfloat).to("cuda:0")
      fft_in_cols = torch.zeros([len(x), len(x)]).to(torch.cfloat).to("cuda:0")
      weights = self.real.state_dict()["weight"] + 1j* self.imag.state_dict()["weight"]
      for i in range(len(x)):
          row_fft_real = self.real(x[i,:]) 
          row_fft_imag = self.imag(x[i,:])
          row_fft = row_fft_real + 1j*row_fft_imag
          fft_in_rows[i,:] = fft_in_rows[i,:] + row_fft
      for i in range(len(x)):
          col_fft = matmul_complex(fft_in_rows[:,i],weights)
          fft_in_cols[:,i] = fft_in_cols[:,i] + col_fft
      
      res = torch.cat([fft_in_cols.real,fft_in_cols.imag],dim=1).to("cuda:0")
      return res
