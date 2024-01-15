import torch
import numpy as np
from types import SimpleNamespace
from utils.utils import *
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import sys
sys.path.append("..")
from utils.effort import *
import tqdm

# Approximation of Q-function given by López-Benítez & Casadevall (2011) based on a second-order exponential function & Q(x) = 1- Q(-x):
# That approximation works for only x>0
a = 0.4920
b = 0.2887
c = 1.1893
Q_function = lambda x: torch.exp(-a*x**2 - b*x - c) 

def Huber(x, delta):
    '''
    Huber function implementation
    '''
    if x.abs()<delta:
        return (x**2)/2
    return delta*(x.abs()-delta/2)

def grad_Huber(x, delta):
    '''
    Gradient of Huber function implementation
    '''
    if x.abs()>delta:
        if x>0:
            return delta
        else:
            return -delta
    return x

def CDF_tau(Yhat, h=0.01, tau=0.5):
    '''
    Approximation of CDF of Gaussian based on the approximate Q function 
    '''
    m = len(Yhat)
    Y_tilde = (tau-Yhat)/h
    sum_ = torch.sum(Q_function(Y_tilde[Y_tilde>0])) \
           + torch.sum(1-Q_function(torch.abs(Y_tilde[Y_tilde<0]))) \
           + 0.5*(len(Y_tilde[Y_tilde==0]))
    return sum_/m

def trainer_kde_fair(model, dataset, optimizer, device, n_epochs, batch_size, z_blind, fairness, lambda_, h, delta_huber, optimal_effort=False, delta_effort=1, effort_iter=20, effort_lr=1, effort_norm='inf'):
    '''
    Training function for KDE method
    '''

    train_tensors, test_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = train_tensors
    X_test, Y_test, Z_test, XZ_test = test_tensors

    sensitive_attrs = dataset.sensitive_attrs
    if z_blind is True:
        train_dataset = FairnessDataset(X_train, Y_train, Z_train)
        test_dataset = FairnessDataset(X_test, Y_test, Z_test)
    else:
        train_dataset = FairnessDataset(XZ_train, Y_train, Z_train)
        test_dataset = FairnessDataset(XZ_test, Y_test, Z_test)
    
    train_dataset, val_dataset = random_split(train_dataset,[int(0.8*len(train_dataset)),len(train_dataset)-int(0.8*len(train_dataset))])
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    tau = 0.5
    pi = torch.tensor(np.pi).to(device)
    phi = lambda x: torch.exp(-0.5*x**2)/torch.sqrt(2*pi)

    results = SimpleNamespace()

    loss_func = torch.nn.BCELoss()

    p_losses = []
    f_losses = []

    dp_disparities = []
    eo_disparities = []
    eodd_disparities = []
    ei_disparities = []
    be_disparities = []
    accuracies = []

    for epoch in tqdm.trange(n_epochs, desc="Training", unit="epochs"):
        
        local_p_loss = []
        local_f_loss = []

        for _, (x_batch, y_batch, z_batch) in enumerate(train_loader):
            x_batch, y_batch, z_batch = x_batch.to(device), y_batch.to(device), z_batch.to(device)
            Yhat = model(x_batch)
            
            cost = 0

            # prediction loss
            p_loss = loss_func(Yhat.reshape(-1), y_batch)
            cost += (1-lambda_)*p_loss

            f_loss = 0
            # EI_Constraint (Equal Improvability)
            if fairness == 'EI' and torch.sum(Yhat<tau)>0:
                x_batch_e = x_batch[(Yhat<tau).reshape(-1),:]
                z_batch_e = z_batch[(Yhat<tau).reshape(-1)]
                if optimal_effort is True:
                    Yhat_max = Optimal_effort(model, dataset, x_batch_e, delta_effort, effort_norm, device=device)
                else:
                    Yhat_max = PGD_effort(model, dataset, x_batch_e, effort_iter, effort_lr, delta_effort, device=device)
                Pr_Ytilde1 = CDF_tau(Yhat_max.detach(), h, tau)

                for sensitive_attr_idx, sensitive_attr in enumerate(sensitive_attrs):
                    for z in sensitive_attr:
                        if torch.sum(z_batch_e[:, sensitive_attr_idx]==z)==0: # we have actually coded the 4 separate groups here
                            continue
                        Pr_Ytilde1_Z = CDF_tau(Yhat_max.detach()[z_batch_e[:, sensitive_attr_idx]==z],h,tau)
                        m_z = z_batch_e[z_batch_e[:, sensitive_attr_idx]==z].shape[0]
                        m = z_batch_e[:, sensitive_attr_idx].shape[0]

                        Delta_z = Pr_Ytilde1_Z-Pr_Ytilde1
                        Delta_z_grad = torch.dot(phi((tau-Yhat_max.detach()[z_batch_e[:, sensitive_attr_idx]==z])/h).view(-1), 
                                                Yhat_max[z_batch_e[:, sensitive_attr_idx]==z].view(-1))/h/m_z
                        Delta_z_grad -= torch.dot(phi((tau-Yhat_max.detach())/h).view(-1), 
                                                Yhat_max.view(-1))/h/m

                        Delta_z_grad *= grad_Huber(Delta_z, delta_huber)
                        f_loss += Delta_z_grad

            cost += lambda_*f_loss
            
            optimizer.zero_grad()
            if (torch.isnan(cost)).any():
                continue
            cost.backward()
            optimizer.step()

            local_p_loss.append(p_loss.item())
            if hasattr(f_loss,'item'):
                local_f_loss.append(f_loss.item())
            else:
                local_f_loss.append(f_loss)

        p_losses.append(np.mean(local_p_loss))
        f_losses.append(np.mean(local_f_loss))

        Yhat_train = model(train_dataset.dataset.X[train_dataset.indices]).reshape(-1).detach().cpu().numpy()
        if optimal_effort is True:
            Yhat_max_train = Optimal_effort(model, dataset, train_dataset.dataset.X[train_dataset.indices], delta_effort, effort_norm, device=device)
        else:
            Yhat_max_train = PGD_effort(model, dataset, train_dataset.dataset.X[train_dataset.indices], effort_iter, effort_lr, delta_effort, device=device)
        Yhat_max_train = Yhat_max_train.reshape(-1).detach().cpu().numpy()

        accuracy, dp_disparity, eo_disparity, eodd_disparity, ei_disparity, be_disparity = model_performance(train_dataset.dataset.Y[train_dataset.indices].detach().cpu().numpy(),
                                                                                                             train_dataset.dataset.Z[train_dataset.indices].detach().cpu().numpy(), 
                                                                                                             Yhat_train,
                                                                                                             Yhat_max_train,
                                                                                                             tau)
        accuracies.append(accuracy)
        dp_disparities.append(dp_disparity)
        eo_disparities.append(eo_disparity)
        eodd_disparities.append(eodd_disparity)
        ei_disparities.append(ei_disparity)
        be_disparities.append(be_disparity)

    results.train_acc_hist = accuracies
    results.train_p_loss_hist = p_losses
    results.train_f_loss_hist = f_losses
    results.train_dp_hist = dp_disparities      
    results.train_eo_hist = eo_disparities  
    results.train_eodd_hist = eodd_disparities  
    results.train_ei_hist = ei_disparities
    results.train_be_hist = be_disparities            

    Yhat_val = model(val_dataset.dataset.X[val_dataset.indices]).reshape(-1).detach().cpu().numpy()
    if optimal_effort is True:
        Yhat_max_val = Optimal_effort(model, dataset, val_dataset.dataset.X[val_dataset.indices], delta_effort, device=device)
    else:
        Yhat_max_val = PGD_effort(model, dataset, val_dataset.dataset.X[val_dataset.indices], effort_iter, effort_lr, delta_effort, device=device)
    Yhat_max_val = Yhat_max_val.reshape(-1).detach().cpu().numpy()
    results.val_acc, results.val_dp, results.val_eo, results.val_eodd, results.val_ei, results.val_be = model_performance(val_dataset.dataset.Y[val_dataset.indices].detach().cpu().numpy(),
                                                                                                                          val_dataset.dataset.Z[val_dataset.indices].detach().cpu().numpy(), 
                                                                                                                          Yhat_val, 
                                                                                                                          Yhat_max_val,
                                                                                                                          tau)
    
    Yhat_test = model(test_dataset.X).reshape(-1).detach().cpu().numpy()
    if optimal_effort is True:
        Yhat_max_test = Optimal_effort(model, dataset, test_dataset.X, delta_effort, effort_norm, device=device)
    else:
        Yhat_max_test = PGD_effort(model, dataset, test_dataset.X, effort_iter, effort_lr, delta_effort, device=device)
    Yhat_max_test = Yhat_max_test.reshape(-1).detach().cpu().numpy()
    results.test_acc, results.test_dp, results.test_eo, results.test_eodd, results.test_ei, results.test_be = model_performance(Y_test.detach().cpu().numpy(), 
                                                                                                                                Z_test.detach().cpu().numpy(), 
                                                                                                                                Yhat_test, 
                                                                                                                                Yhat_max_test, 
                                                                                                                                tau)

    return results

def trainer_fb_fair(model, dataset, optimizer, device, n_epochs, batch_size, z_blind, fairness, lambda_, optimal_effort=False, delta_effort=1, effort_iter=20, effort_lr=1, effort_norm='inf'):
    '''
    Training function for fairbatch method
    '''

    train_tensors, test_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = train_tensors
    X_test, Y_test, Z_test, XZ_test = test_tensors

    sensitive_attrs = dataset.sensitive_attrs
    if z_blind is True:
        train_dataset = FairnessDataset(X_train, Y_train, Z_train)
        test_dataset = FairnessDataset(X_test, Y_test, Z_test)
    else:
        train_dataset = FairnessDataset(XZ_train, Y_train, Z_train)
        test_dataset = FairnessDataset(XZ_test, Y_test, Z_test)
    
    train_dataset, val_dataset = random_split(train_dataset,[int(0.8*len(train_dataset)),len(train_dataset)-int(0.8*len(train_dataset))])
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    tau = 0.5
    pi = torch.tensor(np.pi).to(device)
    phi = lambda x: torch.exp(-0.5*x**2)/torch.sqrt(2*pi)

    results = SimpleNamespace()
    loss_func = torch.nn.BCELoss(reduction = 'mean')

    p_losses = []
    f_losses = []

    dp_disparities = []
    eo_disparities = []
    eodd_disparities = []
    ei_disparities = []
    be_disparities = []
    accuracies = []

    for epoch in tqdm.trange(n_epochs, desc="Training", unit="epochs"):
        
        local_p_loss = []
        local_f_loss = []

        for _, (x_batch, y_batch, z_batch) in enumerate(train_loader):
            x_batch, y_batch, z_batch = x_batch.to(device), y_batch.to(device), z_batch.to(device)
            Yhat = model(x_batch)
            
            cost = 0

            # prediction loss
            p_loss = loss_func(Yhat.reshape(-1), y_batch)
            cost += (1-lambda_)*p_loss

            f_loss = 0

            # EI_Constraint (Equal Improvability)
            if fairness == 'EI' and torch.sum(Yhat<tau)>0:
                x_batch_e = x_batch[(Yhat<tau).reshape(-1),:]
                z_batch_e = z_batch[(Yhat<tau).reshape(-1)]
                if optimal_effort is True:
                    Yhat_max = Optimal_effort(model, dataset, x_batch_e, delta_effort, effort_norm, device=device)
                else:
                    Yhat_max = PGD_effort(model, dataset, x_batch_e, effort_iter, effort_lr, delta_effort, device=device)

                loss_mean = loss_func(Yhat_max.reshape(-1), torch.ones(len(Yhat_max), device=device))
                loss_z = torch.zeros_like(torch.Tensor(sensitive_attrs), device=device)
                for sensitive_attr_idx, sensitive_attr in enumerate(sensitive_attrs):
                    for z in sensitive_attr:
                        z = int(z)
                        group_idx = z_batch_e[:, sensitive_attr_idx] == z
                        if group_idx.sum() == 0:
                            continue
                        loss_z[sensitive_attr_idx][z] = loss_func(Yhat_max.reshape(-1)[group_idx], torch.ones(group_idx.sum(), device=device))
                        f_loss += torch.abs(loss_z[sensitive_attr_idx][z] - loss_mean)
                    
            elif fairness == 'BE':
                x_batch_e = x_batch[(Yhat<tau).reshape(-1),:]
                z_batch_e = z_batch[(Yhat<tau).reshape(-1)]
                if optimal_effort is True:
                    Yhat_max = Optimal_effort(model, dataset, x_batch_e, delta_effort, effort_norm, device=device)
                else:
                    Yhat_max = PGD_effort(model, dataset, x_batch_e, effort_iter, effort_lr, delta_effort, device=device)

                loss_mean = (len(x_batch_e)/len(x_batch)) * loss_func(Yhat_max.reshape(-1), torch.ones(len(Yhat_max), device=device))
                loss_z = torch.zeros_like(torch.Tensor(sensitive_attrs), device=device)
                for sensitive_attr_idx, sensitive_attr in enumerate(sensitive_attrs):
                    for z in sensitive_attr:
                        z = int(z)
                        group_idx = z_batch_e[:, sensitive_attr_idx] == z
                        if group_idx.sum() == 0:
                            continue
                        loss_z[sensitive_attr_idx][z] = (z_batch_e[z_batch_e==z].shape[0]/z_batch[z_batch==z].shape[0])*loss_func(Yhat_max.reshape(-1)[group_idx], torch.ones(group_idx.sum(), device=device))
                        f_loss += torch.abs(loss_z[sensitive_attr_idx][z] - loss_mean)

            cost += lambda_*f_loss
            
            optimizer.zero_grad()
            if (torch.isnan(cost)).any():
                continue
            cost.backward()
            optimizer.step()

            local_p_loss.append(p_loss.item())
            if hasattr(f_loss,'item'):
                local_f_loss.append(f_loss.item())
            else:
                local_f_loss.append(f_loss)

        p_losses.append(np.mean(local_p_loss))
        f_losses.append(np.mean(local_f_loss))

        Yhat_train = model(train_dataset.dataset.X[train_dataset.indices]).reshape(-1).detach().cpu().numpy()
        if optimal_effort is True:
            Yhat_max_train = Optimal_effort(model, dataset, train_dataset.dataset.X[train_dataset.indices], delta_effort, effort_norm, device=device)
        else:
            Yhat_max_train = PGD_effort(model, dataset, train_dataset.dataset.X[train_dataset.indices], effort_iter, effort_lr, delta_effort, device=device)
        Yhat_max_train = Yhat_max_train.reshape(-1).detach().cpu().numpy()

        accuracy, dp_disparity, eo_disparity, eodd_disparity, ei_disparity, be_disparity = model_performance(train_dataset.dataset.Y[train_dataset.indices].detach().cpu().numpy(), train_dataset.dataset.Z[train_dataset.indices].detach().cpu().numpy(), Yhat_train, Yhat_max_train, tau)
        accuracies.append(accuracy)
        dp_disparities.append(dp_disparity)
        eo_disparities.append(eo_disparity)
        eodd_disparities.append(eodd_disparity)
        ei_disparities.append(ei_disparity)
        be_disparities.append(be_disparity)

    results.train_acc_hist = accuracies
    results.train_p_loss_hist = p_losses
    results.train_f_loss_hist = f_losses
    results.train_dp_hist = dp_disparities      
    results.train_eo_hist = eo_disparities  
    results.train_eodd_hist = eodd_disparities  
    results.train_ei_hist = ei_disparities  
    results.train_be_hist = be_disparities       

    Yhat_val = model(val_dataset.dataset.X[val_dataset.indices]).reshape(-1).detach().cpu().numpy()
    if optimal_effort is True:
        Yhat_max_val = Optimal_effort(model, dataset, val_dataset.dataset.X[val_dataset.indices], delta_effort, effort_norm, device=device)
    else:
        Yhat_max_val = PGD_effort(model, dataset, val_dataset.dataset.X[val_dataset.indices], effort_iter, effort_lr, delta_effort, device=device)
    Yhat_max_val = Yhat_max_val.reshape(-1).detach().cpu().numpy()
    results.val_acc, results.val_dp, results.val_eo, results.val_eodd, results.val_ei, results.val_be = model_performance(val_dataset.dataset.Y[val_dataset.indices].detach().cpu().numpy(), val_dataset.dataset.Z[val_dataset.indices].detach().cpu().numpy(), Yhat_val, Yhat_max_val, tau)
    
    Yhat_test = model(test_dataset.X).reshape(-1).detach().cpu().numpy()
    if optimal_effort is True:
        Yhat_max_test = Optimal_effort(model, dataset, test_dataset.X, delta_effort, effort_norm, device=device)
    else:
        Yhat_max_test = PGD_effort(model, dataset, test_dataset.X, effort_iter, effort_lr, delta_effort, device=device)
    Yhat_max_test = Yhat_max_test.reshape(-1).detach().cpu().numpy()
    results.test_acc, results.test_dp, results.test_eo, results.test_eodd, results.test_ei, results.test_be = model_performance(Y_test.detach().cpu().numpy(), Z_test.detach().cpu().numpy(), Yhat_test, Yhat_max_test, tau)

    return results

def trainer_fc_fair(model, dataset, optimizer, device, n_epochs, batch_size, z_blind, fairness, lambda_, optimal_effort=False, delta_effort=1, effort_iter=20, effort_lr=1, effort_norm='inf'):
    '''
    Training function for FC method
    '''

    train_tensors, test_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = train_tensors
    X_test, Y_test, Z_test, XZ_test = test_tensors

    sensitive_attrs = dataset.sensitive_attrs
    if z_blind is True:
        train_dataset = FairnessDataset(X_train, Y_train, Z_train)
        test_dataset = FairnessDataset(X_test, Y_test, Z_test)
    else:
        train_dataset = FairnessDataset(XZ_train, Y_train, Z_train)
        test_dataset = FairnessDataset(XZ_test, Y_test, Z_test)
    
    train_dataset, val_dataset = random_split(train_dataset,[int(0.8*len(train_dataset)),len(train_dataset)-int(0.8*len(train_dataset))])
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    tau = 0.5
    pi = torch.tensor(np.pi).to(device)
    phi = lambda x: torch.exp(-0.5*x**2)/torch.sqrt(2*pi)

    results = SimpleNamespace()

    loss_func = torch.nn.BCELoss()

    p_losses = []
    f_losses = []

    dp_disparities = []
    eo_disparities = []
    eodd_disparities = []
    ei_disparities = []
    be_disparities = []
    accuracies = []

    for epoch in tqdm.trange(n_epochs, desc="Training", unit="epochs"):
        
        local_p_loss = []
        local_f_loss = []

        for _, (x_batch, y_batch, z_batch) in enumerate(train_loader):
            x_batch, y_batch, z_batch = x_batch.to(device), y_batch.to(device), z_batch.to(device)
            Yhat = model(x_batch)
            
            cost = 0

            # prediction loss
            p_loss = loss_func(Yhat.reshape(-1), y_batch)
            cost += (1-lambda_)*p_loss

            f_loss = 0

            # EI_Constraint (Equal Improvability)
            if fairness == 'EI' and torch.sum(Yhat<tau)>0:
                x_batch_e = x_batch[(Yhat<tau).reshape(-1),:]
                z_batch_e = z_batch[(Yhat<tau).reshape(-1)]
                if optimal_effort is True:
                    Yhat_max = Optimal_effort(model, dataset, x_batch_e, delta_effort, effort_norm, device=device)
                else:
                    Yhat_max = PGD_effort(model, dataset, x_batch_e, effort_iter, effort_lr, delta_effort, device=device)
                
                # Handle multiple features
                f_loss += torch.square(torch.mean((z_batch_e-z_batch_e.mean())*Yhat_max))

            cost += lambda_*f_loss
            
            optimizer.zero_grad()
            if (torch.isnan(cost)).any():
                continue
            cost.backward()
            optimizer.step()

            local_p_loss.append(p_loss.item())
            if hasattr(f_loss,'item'):
                local_f_loss.append(f_loss.item())
            else:
                local_f_loss.append(f_loss)

        p_losses.append(np.mean(local_p_loss))
        f_losses.append(np.mean(local_f_loss))

        Yhat_train = model(train_dataset.dataset.X[train_dataset.indices]).reshape(-1).detach().cpu().numpy()
        if optimal_effort is True:
            Yhat_max_train = Optimal_effort(model, dataset, train_dataset.dataset.X[train_dataset.indices], delta_effort, effort_norm, device=device)
        else:
            Yhat_max_train = PGD_effort(model, dataset, train_dataset.dataset.X[train_dataset.indices], effort_iter, effort_lr, delta_effort, device=device)
        Yhat_max_train = Yhat_max_train.reshape(-1).detach().cpu().numpy()

        accuracy, dp_disparity, eo_disparity, eodd_disparity, ei_disparity, be_disparity = model_performance(train_dataset.dataset.Y[train_dataset.indices].detach().cpu().numpy(), train_dataset.dataset.Z[train_dataset.indices].detach().cpu().numpy(), Yhat_train, Yhat_max_train, tau)
        accuracies.append(accuracy)
        dp_disparities.append(dp_disparity)
        eo_disparities.append(eo_disparity)
        eodd_disparities.append(eodd_disparity)
        ei_disparities.append(ei_disparity)
        be_disparities.append(be_disparity)

    results.train_acc_hist = accuracies
    results.train_p_loss_hist = p_losses
    results.train_f_loss_hist = f_losses
    results.train_dp_hist = dp_disparities      
    results.train_eo_hist = eo_disparities  
    results.train_eodd_hist = eodd_disparities  
    results.train_ei_hist = ei_disparities 
    results.train_be_hist = be_disparities        

    Yhat_val = model(val_dataset.dataset.X[val_dataset.indices]).reshape(-1).detach().cpu().numpy()
    if optimal_effort is True:
        Yhat_max_val = Optimal_effort(model, dataset, val_dataset.dataset.X[val_dataset.indices], delta_effort, effort_norm, device=device)
    else:
        Yhat_max_val = PGD_effort(model, dataset, val_dataset.dataset.X[val_dataset.indices], effort_iter, effort_lr, delta_effort, device=device)
    Yhat_max_val = Yhat_max_val.reshape(-1).detach().cpu().numpy()
    results.val_acc, results.val_dp, results.val_eo, results.val_eodd, results.val_ei, results.val_be = model_performance(val_dataset.dataset.Y[val_dataset.indices].detach().cpu().numpy(), val_dataset.dataset.Z[val_dataset.indices].detach().cpu().numpy(), Yhat_val, Yhat_max_val, tau)
    
    Yhat_test = model(test_dataset.X).reshape(-1).detach().cpu().numpy()
    if optimal_effort is True:
        Yhat_max_test = Optimal_effort(model, dataset, test_dataset.X, delta_effort, effort_norm, device=device)
    else:
        Yhat_max_test = PGD_effort(model, dataset, test_dataset.X, effort_iter, effort_lr, delta_effort, device=device)
    Yhat_max_test = Yhat_max_test.reshape(-1).detach().cpu().numpy()
    results.test_acc, results.test_dp, results.test_eo, results.test_eodd, results.test_ei, results.test_be = model_performance(Y_test.detach().cpu().numpy(), Z_test.detach().cpu().numpy(), Yhat_test, Yhat_max_test, tau)

    return results
