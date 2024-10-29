import torch
import torch.nn as nn
import torch.nn.functional as F

class RDBLayer(nn.Module):
    def __init__(self, num_feat=64):
        super(RDBLayer, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        out1 = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        out2 = F.leaky_relu(self.conv2(out1), negative_slope=0.2, inplace=True)
        out3 = F.leaky_relu(self.conv3(out2), negative_slope=0.2, inplace=True)
        out4 = F.leaky_relu(self.conv4(out3), negative_slope=0.2, inplace=True)
        out5 = self.conv5(out4)
        return out5 + x

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64):
        super(RRDBNet, self).__init__()
        
        # первый слой
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # основная часть сети
        self.body = nn.ModuleList()
        # судя по ключам, у нас только один блок в body (body.0)
        rdb_block = nn.ModuleDict({
            'rdb1': RDBLayer(num_feat),
            'rdb2': RDBLayer(num_feat),
            'rdb3': RDBLayer(num_feat)
        })
        self.body.append(rdb_block)

        # последний слой
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        
        body_feat = feat
        for block in self.body:
            body_feat = block['rdb1'](body_feat)
            body_feat = block['rdb2'](body_feat)
            body_feat = block['rdb3'](body_feat)
        
        out = self.conv_last(body_feat)
        return out