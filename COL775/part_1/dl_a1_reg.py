import torch
import torch.nn as nn

class NoNorm(nn.Module):
    def __init__(self):
        super(NoNorm, self).__init__()

    def forward(self, x):
        return x

class BatchNorm(nn.Module):
    def __init__(self, num_features, ep=1e-5, m=0.1):
        super(BatchNorm, self).__init__()
        self.m = m
        self.r_mean = 0
        self.r_var = 0
        self.ep = torch.tensor(ep)
        self.num_features = num_features
        shape = (1, self.num_features, 1, 1)

        self.gamma = nn.Parameter(torch.empty(shape))
        self.beta = nn.Parameter(torch.empty(shape))

        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)
        

    def forward(self, x):

        if not self.training:            
            mean = self.r_mean
            var = self.r_var
        else:
            n = x.numel() / x.size(1)
            dims = (0,2,3)
            var = x.var(dim=dims, keepdim=True, unbiased=False)
            mean = x.mean(dim=dims, keepdim=True)

            with torch.no_grad():
                
                self.r_mean = self.m * mean + (1 - self.m) * self.r_mean
                self.r_var = self.m * (n/(n-1)) * var + (1 - self.m) * self.r_var
        
        bot = torch.sqrt(var + self.ep)
        x = (x - mean)/ bot
        x = x * self.gamma + self.beta

        return x
    
class LayerNorm(nn.Module):
    def __init__(self, num_features, ep=1e-5):
        super(LayerNorm, self).__init__()
        self.ep = torch.tensor(ep)
        self.num_features = num_features
        shape = (1, self.num_features, 1, 1)

        self.gamma = nn.Parameter(torch.empty(shape))
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    def forward(self, x):
        
        _, C, _, _ = x.shape
        assert C == self.num_features
        dims = (1,2,3)
        mean = x.mean(dim=dims, keepdim=True)            
        var = x.var(dim=dims, keepdim=True)

        bot = torch.sqrt(var + self.ep)
        x = (x - mean)/ bot

        x = x * self.gamma + self.beta

        return x
    
class InstanceNorm(nn.Module):
    def __init__(self, num_features, ep=1e-5, affine=True):
        super(InstanceNorm, self).__init__()
        self.r_mean = 0
        self.r_var = 0
        self.ep = torch.tensor(ep)
        self.num_features = num_features
        self.affine = affine
        shape = (1, self.num_features, 1, 1)

        self.gamma = nn.Parameter(torch.empty(shape))
        self.beta = nn.Parameter(torch.empty(shape))
            
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)


    def forward(self, x):
        _, C, _, _ = x.shape

        assert C == self.num_features
        dims = (2,3)
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True)
        dn = torch.sqrt(var + self.ep)
        x = (x - mean)/ dn
        
        if self.affine:
            x = x * self.gamma + self.beta

        return x


class GroupNorm(nn.Module):
    def __init__(self, num_features, ep=1e-5, group=4):
        super(GroupNorm,self).__init__()
        self.ep = torch.tensor(ep)
        self.num_features = num_features
        self.group = group        
        shape = (1, self.num_features, 1, 1)

        self.gamma = nn.Parameter(torch.empty(shape))
        self.beta = nn.Parameter(torch.empty(shape))
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    def forward(self, x):
        N, C, H, W = x.shape

        assert C % self.group == 0
        assert self.num_features == C

        x = x.view(N, self.group, int(C / self.group), H, W)
        dims = (1,2,3)
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True)
        dn = torch.sqrt(var + self.ep)
        x = (x - mean)/ dn
        x = x.view(N, C, H, W)
        
        x = x * self.gamma + self.beta

        return x
    
class BatchInstanceNorm(nn.Module):
    def __init__(self, num_features, m = 0.1, ep=1e-5, rho=0.5):
        super(BatchInstanceNorm, self).__init__()
        self.m = m
        self.r_mean = 0
        self.r_var = 0
        self.ep = torch.tensor(ep)
        self.num_features = num_features
        self.rho = rho
        shape = (1, self.num_features, 1, 1)
        self.gamma = nn.Parameter(torch.empty(shape))
        self.beta = nn.Parameter(torch.empty(shape))
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    
    def forward(self, x):
        if not self.training:            
            mean_bn = self.r_mean
            var_bn = self.r_var
        else:
            n = x.numel() / x.size(1)
            dims = (0,2,3)
            var_bn = x.var(dim=dims, keepdim=True, unbiased=False)
            mean_bn = x.mean(dim=dims, keepdim=True)

            with torch.no_grad():
                
                self.r_mean = self.m * mean_bn + (1 - self.m) * self.r_mean
                self.r_var = self.m * (n/(n-1)) * var_bn + (1 - self.m) * self.r_var

        dn = torch.sqrt(var_bn + self.ep)
        x_bn = (x - mean_bn)/ dn
        dims = (2,3)
        mean_in = x.mean(dim=dims, keepdim=True)
        var_in = x.var(dim=dims, keepdim=True)
        dn = torch.sqrt(var_in + self.ep)
        x_in = (x - mean_in)/ dn

        x = self.rho * x_bn + (1-self.rho) * x_in
        x = x * self.gamma + self.beta

        return x
    



