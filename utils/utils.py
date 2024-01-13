from torch.utils.data import Dataset
import numpy as np

class FairnessDataset(Dataset):
    '''
    An abstract dataset class wrapped around Pytorch Dataset class.
    
    Dataset consists of 3 parts; X, Y, Z.
    '''
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x, y, z = self.X[index], self.Y[index], self.Z[index]
        return x, y, z

def DPDisparity(n_yz, each_z = False):
    '''
    Demographic disparity: max_z{|P(yhat=1|z=z)-P(yhat=1)|}

    Parameters
    ----------
    n_yz : dictionary
        #(yhat=y,z=z)
    each_z : Bool
        Returns for each sensitive group or max
    '''
    z_set = list(set([z for _, z in n_yz.keys()]))
    
    dp = []
    p11 = sum([n_yz[(1,z)] for z in z_set]) / sum([n_yz[(1,z)]+n_yz[(0,z)] for z in z_set])
    for z in z_set:
        try:
            dp_z = abs(n_yz[(1,z)]/(n_yz[(0,z)] + n_yz[(1,z)]) - p11)
        except ZeroDivisionError:
            if n_yz[(1,z)] == 0: 
                dp_z = 0
            else:
                dp_z = 1
        dp.append(dp_z)
    if each_z:
        return dp
    else:
        return max(dp)

def EIDisparity(n_eyz, each_z = False):
    '''
    Equal improvability disparity: max_z{|P(yhat_max=1|z=z,y_hat=0)-P(yhat_max=1|y_hat=0)|}

    Parameters
    ----------
    n_eyz : dictionary
        #(yhat_max=e,yhat=y,z=z)
    each_z : Bool
        Returns for each sensitive group or max
    '''
    z_set = list(set([z for _,_, z in n_eyz.keys()]))
    
    ei = []
    if sum([n_eyz[(1,0,z)]+n_eyz[(0,0,z)] for z in z_set])==0:
        p10 = 0
    else:
        p10 = sum([n_eyz[(1,0,z)] for z in z_set]) / sum([n_eyz[(1,0,z)]+n_eyz[(0,0,z)] for z in z_set])
    for z in z_set:
        if n_eyz[(1,0,z)] == 0: 
            ei_z = 0
        else:
            ei_z = n_eyz[(1,0,z)]/(n_eyz[(0,0,z)] + n_eyz[(1,0,z)])
        ei.append(abs(ei_z-p10))
    if each_z:
        return ei
    else:
        return max(ei)

def BEDisparity(n_eyz, each_z = False):
    '''
    Bounded effort disparity: max_z{|P(yhat_max=1,y_hat=0|z=z)-P(yhat_max=1,y_hat=0)|}

    Parameters
    ----------
    n_eyz : dictionary
        #(yhat_max=e,yhat=y,z=z)
    each_z : Bool
        Returns for each sensitive group or max
    '''
    z_set = list(set([z for _,_, z in n_eyz.keys()]))
    
    be = []
    if sum([n_eyz[(1,0,z)]+n_eyz[(0,0,z)]+n_eyz[(1,1,z)]+n_eyz[(0,1,z)] for z in z_set])==0:
        p1 = 0
    else:
        p1 = sum([n_eyz[(1,0,z)] for z in z_set]) / sum([n_eyz[(1,0,z)]+n_eyz[(0,0,z)]+n_eyz[(1,1,z)]+n_eyz[(0,1,z)] for z in z_set])
    for z in z_set:
        if n_eyz[(1,0,z)] == 0: 
            be_z = 0
        else:
            be_z = n_eyz[(1,0,z)]/(n_eyz[(1,0,z)]+n_eyz[(0,0,z)]+n_eyz[(1,1,z)]+n_eyz[(0,1,z)])
        be.append(abs(be_z-p1))
    if each_z:
        return be
    else:
        return max(be)

def EODisparity(n_eyz, each_z = False):
    '''
    Equal opportunity disparity: max_z{|P(yhat=1|z=z,y=1)-P(yhat=1|y=1)|}

    Parameters
    ----------
    n_eyz : dictionary
        #(yhat=e,y=y,z=z)
    each_z : Bool
        Returns for each sensitive group or max
    '''
    z_set = list(set([z for _,_, z in n_eyz.keys()]))

    eod = []
    p11 = sum([n_eyz[(1,1,z)] for z in z_set]) / sum([n_eyz[(1,1,z)]+n_eyz[(0,1,z)] for z in z_set])
    for z in z_set:
        try:
            eod_z = abs(n_eyz[(1,1,z)]/(n_eyz[(0,1,z)] + n_eyz[(1,1,z)]) - p11)
        except ZeroDivisionError:
            if n_eyz[(1,1,z)] == 0: 
                eod_z = 0
            else:
                eod_z = 1
        eod.append(eod_z)
    if each_z:
        return eod
    else:
        return max(eod)
    
def EODDDisparity(n_eyz, each_z = False):
    '''
    Equalized odds disparity: max_z_y{|P(yhat=1|z=z,y=y)-P(yhat=1|y=y)|}

    Parameters
    ----------
    n_eyz: dictionary
        #(yhat=e,y=y,z=z)
    each_z : Bool
        Returns for each sensitive group or max
    '''
    z_set = list(set([z for _,_,z in n_eyz.keys()]))
    y_set = list(set([y for _,y,_ in n_eyz.keys()]))
    
    eoddd = []
    for y in y_set:
        p = sum([n_eyz[(1,y,z)] for z in z_set]) / sum([n_eyz[(1,y,z)]+n_eyz[(0,y,z)] for z in z_set])
        for z in z_set:
            try:
                eoddd_z = abs(n_eyz[(1,y,z)]/(n_eyz[(0,y,z)] + n_eyz[(1,y,z)]) - p)
            except ZeroDivisionError:
                if n_eyz[(1,y,z)] == 0: 
                    eoddd_z = 0
                else:
                    eoddd_z = 1
            eoddd.append(eoddd_z)
    if each_z:
        return eoddd
    else:
        return max(eoddd)

def model_performance(Y, Z, Yhat, Yhat_max, tau):
    Ypred = (Yhat>tau)*1
    Ypred_max = (Yhat_max>tau)*1
    acc = np.mean(Y==Ypred)

    n_yz = {}
    n_eyz = {}
    n_mez = {}

    for y_hat in [0,1]:
        for y in [0,1]:
            for column in range(Z.shape[1]):
                for z in np.unique(Z[:, column]):
                    n_eyz[(y_hat, y, (column, z))] = np.sum((Ypred==y_hat)*(Y==y)*(Z[:, column]==z))
                    n_mez[(y_hat, y, (column, z))] = np.sum((Ypred_max==y_hat)*(Ypred==y)*(Z[:, column]==z))
    
    for y in [0,1]:
        for column in range(Z.shape[1]):
            for z in np.unique(Z[:, column]):
                n_yz[(y, (column, z))] = np.sum((Ypred==y)*(Z[:, column]==z))

    return acc, DPDisparity(n_yz), EODisparity(n_eyz), EODDDisparity(n_eyz), EIDisparity(n_mez), BEDisparity(n_mez)
