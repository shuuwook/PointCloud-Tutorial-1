import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from module.dataset import ModelNet40
from module.utils import *

import os, sys
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
init 함수에서 feature_dim을 input으로 받아 Batch Normalization을 수행합니다.
BatchNorm(3), BatchNorm(64), BatchNorm(1024)와 같이 사용됩니다.
"""

class BatchNorm(nn.Module):
    '''
        Perform batch normalization.
        Input: A tensor of size (N, M, feature_dim), or (N, feature_dim, M) (available when feature_dim != M), 
                or (N, feature_dim)
        Output: A tensor of the same size as input.
    '''
    def __init__(self, feature_dim):
        super(BatchNorm, self).__init__()
        

    def forward(self, x):
        
        return self.batchnorm(x)

"""
init 함수에서 받은 tuple형태의 param 파라미터를 통해 torch.permute_(param)을 실행합니다.
예를 들어, x가 (100, 200, 300)의 shape를 갖고 있을 때 x.permute((0, 1, 2))는 x와 동일합니다.
"""

class Permute(nn.Module):
    def __init__(self, param):
        super(Permute, self).__init__()
        self.param = param

    def forward(self, x):
        return x.permute(self.param)

"""
Fully connected layers로 이루어진 MLP를 구성합니다.
일반적으로 Fully connected layer, Batch normalization, Activation function이 set로 구성됩니다.

init 함수에서 hidden_size 파라미터는 
input dimension부터 hidden dimension들, output dimension까지의 tuple로 입력받습니다.

input dimension부터 output dimension까지를 한번에 입력받는다는 점에 주의해주세요.

batchnorm과 last_activation argument에 따라 옵션을 줄 수 있도록 작성하면 좋지만, 무시하셔도 괜찮습니다.
"""

class MLP(nn.Module):
    def __init__(self, hidden_size, batchnorm = True, last_activation = True):
        super(MLP, self).__init__()
        

    def forward(self, x):
        return self.mlp(x)

"""
일반적인 max pooling이 아닌 global maxpooling입니다.
input의 shape가 (B, N, D)일 때, output의 shape가 (B,)
"""

class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, x, dim=1, keepdim = False):
        
        return res

"""
Input 또는 중간의 feature를 permutation과 rigid motion에 invariant하도록 만들기 위한 모듈입니다.
- nfeat, 64, 128, 1024로 mapping되는 MLP
- max pooling, batch normalization
- 다시 1024, 512, 256, nfeat*nfeat로 mapping되는 MLP로 구성됩니다.

최종적으로 (B, n_feat*n_feat)의 output을 (B, n_feat, n_feat)로 shape을 변경해 return해주면 됩니다.
"""

class TNet(nn.Module):
    def __init__(self, nfeat):
        super(TNet, self).__init__()
        
        
    def forward(self, x):
        batch_size = x.shape[0]
        return self.tnet(x).view(batch_size, self.nfeat, self.nfeat)

"""
위의 모듈들을 모두 조합해 최종적으로 point cloud classification을 위한 pointNet을 구성해보겠습니다.
TNet의 output은 transform matrix이므로 TNet의 input과 matrix multiplication을 해야합니다.

- TNet : (B,N,3) > (B,N,3)
- BatchNorm : (B,N,3) > (B,N,3)
- MLP : (B,N,3) > (B,N,64) > (B,N,64)
- TNet : (B,N,64) > (B,N,64)
- BatchNorm : (B,N,64) > (B,N,64)
- MLP : (B,N,64) > (B,N,64) > (B,N,128) > (B,N,1024)
- Maxpooling : (B,N,1024) > (B,1024)
- BatchNorm : (B,1024) > (B,1024)
- MLP : (B,1024) > (B,512) > (B,256)
- Dropout : (B,256) > (B,256)
- FC-layer (Linear or Conv1d) for k classes : (B,256) > (B,nclass)
"""

class PointNet(nn.Module):
    def __init__(self, nfeat, nclass, dropout = 0):
        super(PointNet, self).__init__()
        
        
        self.eye64 = torch.eye(64).to(device)

    def forward(self, xs):
        batch_size = xs.shape[0]
        
        
        if (self.training):
            transform_transpose = transform.transpose(1, 2)
            tmp = torch.stack([torch.mm(transform[i], transform_transpose[i]) for i in range(batch_size)])
            L_reg = ((tmp - self.eye64) ** 2).sum() / batch_size
            
        return (F.log_softmax(xs, dim=1), L_reg) if self.training else F.log_softmax(xs, dim=1)

################### Arguments ###################

lr = 0.001
num_points = 128
save_name = "PointNet.pt"
batch_size = 512

################## loading data ##################

train_data = ModelNet40(num_points)
test_data = ModelNet40(num_points, 'test')

train_size = int(0.9 * len(train_data))
valid_size = len(train_data) - train_size
train_data, valid_data = Data.random_split(train_data, [train_size, valid_size])
valid_data.partition = 'valid'
train_data.partition = 'train'

print("train data size: ", len(train_data))
print("valid data size: ", len(valid_data))
print("test data size: ", len(test_data))

def collate_fn(batch):
    Xs = torch.stack([X for X, _ in batch])
    Ys = torch.tensor([Y for _, Y in batch], dtype = torch.long)
    return Xs, Ys

train_iter  = Data.DataLoader(train_data, shuffle = True, batch_size = batch_size, collate_fn = collate_fn)
valid_iter = Data.DataLoader(valid_data, batch_size = batch_size, collate_fn = collate_fn)
test_iter = Data.DataLoader(test_data, batch_size = batch_size, collate_fn = collate_fn)

############### loading model ####################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = PointNet(nfeat=3, nclass=40, dropout=0.3)
net.to(device)
print(net)

############### training #########################

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay = 0.0001)
loss = nn.NLLLoss()

def adjust_lr(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

retrain = True
if os.path.exists(save_name):
    print("Model parameters have already been trained before. Retrain ? (y/n)")
    ans = input()
    if (ans == 'y'):
        checkpoint = torch.load(save_name, map_location = device)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for g in optimizer.param_groups:
            g['lr'] = lr

train_model(train_iter, valid_iter, net, loss, optimizer, device = device, max_epochs = int(1000/(batch_size/64)), 
            adjust_lr = adjust_lr, early_stop = EarlyStop(patience = 20, save_name = save_name))
    

############### testing ##########################

loss, acc = evaluate_model(test_iter, net, loss)
print('test acc = %.6f' % (acc))
