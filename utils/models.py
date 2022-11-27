import torch.nn as nn

class logReg(nn.Module):
    '''
    Logistic Regression model

    It works only for binary logistic regression. 
    '''
    def __init__(self, num_features):
        '''
        Parameters
        ----------
        num_features : int
            length of feature vector
        '''
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_features,1),
            nn.Sigmoid())

    def forward(self, x):
        '''
        Forward pass for logistic regression

        Parameters
        ----------
        x : Tensor
            Input tensor for the model
        
        Return
        ------
        x : Tensor
            Output of the model
        '''
        x = self.layers(x)
        return x

class MLP(nn.Module):
    '''
    Multilayer Perceptron

    It works only for binary classification. 
    '''
    def __init__(self, num_features, n_layers):
        '''
        Parameters
        ----------
        num_features : int
            length of feature vector
        '''
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(num_features, n_layers[0]))
        layers.append(nn.ReLU())
        for i in range(len(n_layers)-1):
            layers.append(nn.Linear(n_layers[i], n_layers[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_layers[-1],1))
        layers.append(nn.Sigmoid()) 
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward pass for logistic regression

        Parameters
        ----------
        x : Tensor
            Input tensor for the model
        
        Return
        ------
        x : Tensor
            Output of the model
        '''
        x = self.layers(x)
        return x