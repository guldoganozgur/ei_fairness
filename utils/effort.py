from multiprocessing import reduction
import torch
from torch.autograd import Variable

def PGD_effort(model, dataset, x, iter, lr, delta, device="cpu"):
    
    efforts = Variable(torch.zeros(x.shape, device=device), requires_grad = True)
    
    improvable_indices = []
    for i in range(efforts.shape[1]):
        if i not in dataset.U_index:
            improvable_indices.append(i)
    C_min = torch.zeros(x[:,dataset.C_index].shape, device=device)
    C_max = torch.zeros(x[:,dataset.C_index].shape, device=device)
    for j in range(len(dataset.C_index)):        
        C_min[:, j] = dataset.C_min[j]-x[:,dataset.C_index[j]]
        C_max[:, j] = dataset.C_max[j]-x[:,dataset.C_index[j]]
    
    loss = torch.nn.BCELoss(reduction='sum')
    for i in range(iter):
        Yhat = model(x + efforts)
        cost = loss(Yhat.squeeze(),torch.ones(Yhat.squeeze().shape))
        model.zero_grad()
        cost.backward()

        efforts_update = efforts - (lr/((i+1)**.5))*efforts.grad
        efforts_update[:,improvable_indices] = torch.clamp(efforts_update[:,improvable_indices], -delta, delta)

        efforts_update[:,dataset.C_index] = torch.round(efforts_update[:,dataset.C_index])
        efforts_update[:,dataset.C_index] = torch.clamp(efforts_update[:,dataset.C_index], C_min, C_max)
        efforts_update[:,dataset.U_index] = torch.zeros(efforts[:,dataset.U_index].shape)
        efforts = Variable(efforts_update, requires_grad = True)
        
    Yhat = model(x + efforts)

    return Yhat

def Optimal_effort(model, dataset, x, delta, norm='inf', device="cpu"):
    
    efforts = Variable(torch.zeros(x.shape, device=device), requires_grad = True)
    
    improvable_indices = []
    for i in range(efforts.shape[1]):
        if i not in dataset.U_index:
            improvable_indices.append(i)
    C_min = torch.zeros(x[:,dataset.C_index].shape, device=device)
    C_max = torch.zeros(x[:,dataset.C_index].shape, device=device)
    for j in range(len(dataset.C_index)):        
        C_min[:, j] = dataset.C_min[j]-x[:,dataset.C_index[j]]
        C_max[:, j] = dataset.C_max[j]-x[:,dataset.C_index[j]]

    for name, param in model.named_parameters():
        if 'layers.0.weight' in name:
            weights = param.detach() 
    if norm=='inf':
        efforts_update = delta * torch.sign(weights) * torch.ones(x.shape, device=device)
        efforts_update[:,improvable_indices] = torch.clamp(efforts_update[:,improvable_indices], -delta, delta)
    else: # elif norm=='l2':
        efforts_update = delta * (weights / torch.square(torch.sum(weights*weights))) * torch.ones(x.shape, device=device)
    efforts_update[:,dataset.C_index] = torch.round(efforts_update[:,dataset.C_index])
    efforts_update[:,dataset.C_index] = torch.clamp(efforts_update[:,dataset.C_index], C_min, C_max)
    efforts_update[:,dataset.U_index] = torch.zeros(efforts[:,dataset.U_index].shape, device=device)
    efforts = Variable(efforts_update, requires_grad=True)
        
    Yhat = model(x + efforts)

    return Yhat