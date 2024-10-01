import torch
import numpy as np

class TrigFourierSeries():
    def __init__(self, T, fx, nh):

        # T -> time period of the function
        # fx (torch.tensor, 1D) -> equally sampled data in [0,T] of a function f with f(0) = fx[0] = fx[-1] = f(T)
        # nh -> number of harmonics we want to compute

        n = len(fx)
        x = torch.from_numpy(np.linspace(0, T, n))
        fbas = 2*np.pi / T
        aux = 2.0 / (n-1)
        x = x[0: -1]
        fx = fx[0: -1]

        A0d2 = torch.sum(fx) * aux / 2.0
        cov = torch.zeros([n-1, nh])
        siv = torch.zeros([n-1, nh])

        cov[:, 0] = torch.cos(fbas * x)
        siv[:, 0] = torch.sin(fbas * x)

        for k in range(1, nh):
            cov[:, k] = cov[:, k-1] * cov[:, 0] - siv[:, k-1] * siv[:, 0]
            siv[:, k] = siv[:, k-1] * cov[:, 0] + cov[:, k-1] * siv[:, 0]

        Ak = fx @ cov * aux
        Bk = fx @ siv * aux
        pari = 0
        if torch.max(torch.abs(Bk)) < 1e-12:
            pari = 1
        if torch.max(torch.max(torch.abs(A0d2), torch.abs(Ak))) < 1e-12:
            pari = 2

        self.nharm = nh
        self.pari = pari
        self.T = T
        self.A0d2 = A0d2
        self.Ak = Ak
        self.Bk = Bk

    def eval(self, x):
	# Evaluates the fourier series in a domain point 'x'.
        fx = 0
        
        
        for i in range(0,self.nharm):
            fx = fx + self.Ak[i] * torch.cos((i+1) * x * 2 * torch.pi / self.T)  + self.Bk[i] * torch.sin(2* (i+1) * x * torch.pi / self.T)
        return fx + self.A0d2

    
    def evalNoGrad(self, x):
	# Evaluates the fourier series in a domain point 'x'. Should be faster than eval(), but breaks the computation graph.
        nt = len(x)
        fbas = 2 * torch.pi / self.T

        if self.nharm == 0:
            fsx = self.A0d2 * torch.ones((nt, nt))
            return fsx

        cov = torch.zeros((self.nharm, nt))
        siv = torch.zeros((self.nharm, nt))
        cov[0, :] = torch.cos(fbas * x)
        siv[0, :] = torch.sin(fbas * x)

        for k in range(1, self.nharm):
            cov[k,:] = cov[k-1, :] * cov[0, :] - siv[k-1, :] * siv[0, :]
            siv[k,:] = siv[k-1, :] * cov[0, :] + cov[k-1, :] * siv[0, :]

        fsx = torch.zeros( nt)

        if self.pari == 0 or self.pari == 2:
            fsx = fsx + self.Bk @ siv
        if self.pari == 0 or self.pari == 1:
            fsx = fsx + self.Ak @ cov
            fsx = fsx + self.A0d2 
        return fsx

    