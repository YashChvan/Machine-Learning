import torch
import torch.nn as nn

class NoNorm(nn.Module):
    def __init__(self):
        super(NoNorm, self).__init__()

    def forward(self, x):
        return x

class BatchNorm(nn.Module):
    """
    Custom implementation of Batch Normalization for 2D inputs.

    This implementation follows the principles of Batch Normalization but includes modifications
    for better compatibility and efficiency.

    Args:
        num_features (int): Number of channels in the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability.
        momentum (float, optional): The value used for the running_mean and running_var computation.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.scale = nn.Parameter(torch.ones(num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_features, 1, 1))
        self.register_buffer('running_mean', torch.zeros(num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(num_features, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        y = self.scale * x_normalized + self.bias
        return y

    
class LayerNorm(nn.Module):
    def __init__(self, n_fea, para1=1e-5):
        super(LayerNorm, self).__init__()
        self.para1 = torch.tensor(para1)
        self.n_fea = n_fea
        shp = (1, self.n_fea, 1, 1)
        
        self.g1 = nn.Parameter(torch.empty(shp))
        self.b1 = nn.Parameter(torch.empty(shp))
        nn.init.zeros_(self.b1)
        nn.init.ones_(self.g1)

    def forward(self, x):
        
        _, C, _, _ = x.shape
        assert C == self.n_fea
        dims = (1,2,3)
        mean = x.mean(dim=dims, keepdim=True)            
        var = x.var(dim=dims, keepdim=True)

        bot = torch.sqrt(var + self.para1)
        x = (x - mean)/ bot

        x = x * self.g1 + self.b1

        return x
    
class InstanceNorm(nn.Module):
    def __init__(self, n_fea, para1=1e-5):
        super(InstanceNorm, self).__init__()
        self.r_m = 0
        self.r_var = 0
        self.para1 = torch.tensor(para1)
        self.n_fea = n_fea
        shp = (1, self.n_fea, 1, 1)
        self.g1 = nn.Parameter(torch.empty(shp))
        self.b1 = nn.Parameter(torch.empty(shp))
            
        nn.init.zeros_(self.b1)
        nn.init.ones_(self.g1)


    def forward(self, x):
        _, C, _, _ = x.shape

        assert C == self.n_fea
        dims = (2,3)
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True)
        dn = torch.sqrt(var + self.para1)
        x = (x - mean)/ dn
        x = x * self.g1 + self.b1

        return x


class GroupNorm(nn.Module):
    def __init__(self, n_fea, para1=1e-5, group=4):
        super(GroupNorm,self).__init__()
        self.para1 = torch.tensor(para1)
        self.n_fea = n_fea
        self.group = group        
        shp = (1, self.n_fea, 1, 1)
        self.g1 = nn.Parameter(torch.empty(shp))
        self.b1 = nn.Parameter(torch.empty(shp))
        nn.init.zeros_(self.b1)
        nn.init.ones_(self.g1)

    def forward(self, x):
        N, C, H, W = x.shape

        assert C % self.group == 0
        assert self.n_fea == C

        x = x.view(N, self.group, int(C / self.group), H, W)
        dims = (1,2,3)
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True)
        dn = torch.sqrt(var + self.para1)
        x = (x - mean)/ dn
        x = x.view(N, C, H, W)
        x = x * self.g1 + self.b1
        return x
    
class BatchInstanceNorm(nn.Module):
    def __init__(self, n_fea, mnt = 0.1, para1=1e-5, rho=0.5):
        super(BatchInstanceNorm, self).__init__()
        self.mnt = mnt
        self.r_m = 0
        self.r_var = 0
        self.para1 = torch.tensor(para1)
        self.n_fea = n_fea
        self.rho = rho
        shp = (1, self.n_fea, 1, 1)
        self.g1 = nn.Parameter(torch.empty(shp))
        self.b1 = nn.Parameter(torch.empty(shp))
        nn.init.zeros_(self.b1)
        nn.init.ones_(self.g1)

    
    def forward(self, x):
        if not self.training:            
            mean_bn = self.r_m
            var_bn = self.r_var
        else:
            n = x.numel() / x.size(1)
            dims = (0,2,3)
            var_bn = x.var(dim=dims, keepdim=True, unbiased=False)
            mean_bn = x.mean(dim=dims, keepdim=True)

            with torch.no_grad():
                
                self.r_m = self.mnt * mean_bn + (1 - self.mnt) * self.r_m
                self.r_var = self.mnt * (n/(n-1)) * var_bn + (1 - self.mnt) * self.r_var

        dn = torch.sqrt(var_bn + self.para1)
        x_bn = (x - mean_bn)/ dn
        dims = (2,3)
        mean_in = x.mean(dim=dims, keepdim=True)
        var_in = x.var(dim=dims, keepdim=True)
        dn = torch.sqrt(var_in + self.para1)
        x_in = (x - mean_in)/ dn

        x = self.rho * x_bn + (1-self.rho) * x_in
        x = x * self.g1 + self.b1

        return x