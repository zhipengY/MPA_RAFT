import torch
import torch.nn as nn
import torch.nn.functional as F

# from core.opts import motion_att
import math

class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


# class FPResidualBlock(nn.Module):
#     def __init__(self, in_planes, planes, norm_fn='group', stride=1):
#         super(FPResidualBlock, self).__init__()

#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
#         self.conv2 = nn.Conv2d(planes + in_planes, planes, kernel_size=3, padding=1)
#         self.conv2_1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=stride)
#         self.pool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=stride)
#         self.cony3 = nn.Conv2d(planes * 2, planes, kernel_size=1, stride=1)
#         self.attention = simam_module()
#         self.relu = nn.ReLU(inplace=True)

#         num_groups = planes // 8

#         if norm_fn == 'group':
#             self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
#             self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
#             if not stride == 1:
#                 self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

#         elif norm_fn == 'batch':
#             self.norm1 = nn.BatchNorm2d(planes)
#             self.norm2 = nn.BatchNorm2d(planes)
#             if not stride == 1:
#                 self.norm3 = nn.BatchNorm2d(planes)

#         elif norm_fn == 'instance':
#             self.norm1 = nn.InstanceNorm2d(planes)
#             self.norm2 = nn.InstanceNorm2d(planes)
#             if not stride == 1:
#                 self.norm3 = nn.InstanceNorm2d(planes)

#         elif norm_fn == 'none':
#             self.norm1 = nn.Sequential()
#             self.norm2 = nn.Sequential()
#             if not stride == 1:
#                 self.norm3 = nn.Sequential()

#         if stride == 1:
#             self.downsample = None

#         else:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)
            
#         # self.att1 = motion_att(planes)

#     def forward(self, x):
#         y = x
#         # y = self.relu(self.norm1(self.conv1(y)))
#         # y = self.relu(self.norm2(self.conv2(y)))
#         #
#         y1 = self.relu(self.norm1(self.att1(self.conv1(y))))
#         p1 = self.pool1(y)
#         y1 = torch.cat((y1, p1), 1)
#         y2 = self.relu(self.norm2(self.att1(self.conv2(y1))))

#         # p2 = self.pool2(y1)
#         # y2 = self.attention(torch.cat((y2, p2), 1))
        
#         yw = self.relu(self.norm1(self.att1(self.conv1(y))))
#         yw = self.relu(self.norm2(self.att1(self.conv2_1(yw))))

#         # y = self.relu(self.norm2(self.conv2(y) + self.pool2(y)))

#         y3 = torch.cat((yw, y2), 1)

#         y3 = self.cony3(y3)

#         if self.downsample is not None:
#             x = self.downsample(x)

#         return self.relu(x + y3)


# class FPRBasicEncoder(nn.Module):
#     def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
#         super(FPRBasicEncoder, self).__init__()
#         self.norm_fn = norm_fn

#         if self.norm_fn == 'group':
#             self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

#         elif self.norm_fn == 'batch':
#             self.norm1 = nn.BatchNorm2d(64)

#         elif self.norm_fn == 'instance':
#             self.norm1 = nn.InstanceNorm2d(64)

#         elif self.norm_fn == 'none':
#             self.norm1 = nn.Sequential()

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.relu1 = nn.ReLU(inplace=True)

#         self.in_planes = 64
#         self.layer1 = self._make_layer(64, stride=1)
#         self.layer2 = self._make_layer(96, stride=2)
#         self.layer3 = self._make_layer(128, stride=2)

#         # output convolution
#         self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

#         self.dropout = None
#         if dropout > 0:
#             self.dropout = nn.Dropout2d(p=dropout)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
#                 if m.weight is not None:
#                     nn.init.constant_(m.weight, 1)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def _make_layer(self, dim, stride=1):
#         layer1 = FPResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
#         layer2 = FPResidualBlock(dim, dim, self.norm_fn, stride=1)
#         layers = (layer1, layer2)

#         self.in_planes = dim
#         return nn.Sequential(*layers)

#     def forward(self, x):

#         # if input is list, combine batch dimension
#         is_list = isinstance(x, tuple) or isinstance(x, list)
#         if is_list:
#             batch_dim = x[0].shape[0]
#             x = torch.cat(x, dim=0)

#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.relu1(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)

#         x = self.conv2(x)

#         if self.training and self.dropout is not None:
#             x = self.dropout(x)

#         if is_list:
#             x = torch.split(x, [batch_dim, batch_dim], dim=0)

#         return x


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)
         
        # self.att = motion_att(planes)
            
    def forward(self, x):
        b, c, h, w = x.size()
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))
        # y = self.att(y)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class ConvMLP(nn.Module):
    def __init__(self, hidden_dim, kernel_size=1, stride=1, padding=0, norm_fn='batch', dropout=0.0):
        super(ConvMLP, self).__init__()
        
        self.norm_fn = norm_fn
        if self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(hidden_dim)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(hidden_dim)
        elif self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(32, hidden_dim)
        elif self.norm_fn == 'none':
            self.norm1 = None
        
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        return x

class SK_module_1(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, norm_fn='batch', dropout=0):
        super(SK_module_1, self).__init__()

        self.norm_fn = norm_fn
        if self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(in_planes)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(in_planes)
        elif self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(32, out_planes)
        elif self.norm_fn == 'none':
            self.norm1 = None
            
        self.conv1_1x1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.ReLU()
        self.conv2_1 = nn.Conv2d(out_planes, out_planes, kernel_size=(17,1), stride=1, padding=(17//2, 0), bias=False)
        self.conv2_2 = nn.Conv2d(out_planes, out_planes, kernel_size=(1,17), stride=1, padding=(0, 17//2), bias=False)
        self.conv2_1x1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.mlp = ConvMLP(out_planes)
        
    def forward(self, x):
        
        y = x
        
        x = self.conv1_1x1(x)
        x = self.norm1(x)
        x = self.act(x)
        
        x1 = self.conv2_1(x)
        x = x + x1
        # x = self.act(x)
        x2 = self.conv2_2(x1)
        x = x + x2
        x = self.conv2_1x1(x)
        x = x + y
        x = self.mlp(x)
        
        return x



class SK_module_2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, norm_fn='batch', dropout=0):
        super(SK_module_2, self).__init__()

        self.norm_fn = norm_fn
        if self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(in_planes)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(in_planes)
        elif self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(32, in_planes)
        elif self.norm_fn == 'none':
            self.norm1 = None
            
        self.conv1_1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.act = nn.ReLU()
        self.conv1_1_1x1 = nn.Conv2d(2*out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2_1 = nn.Conv2d(out_planes, out_planes, kernel_size=(17,1), stride=1, padding=(17//2, 0), bias=False)
        self.conv2_2 = nn.Conv2d(out_planes, out_planes, kernel_size=(1,17), stride=1, padding=(0, 17//2), bias=False)
        self.conv2_1x1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.mlp = ConvMLP(out_planes)
        
    def forward(self, x):
        
        y = x
        
        x = self.conv1_1x1(x)
        x = self.norm1(x)
        x = self.act(x)
        
        x1 = self.conv2_1(x)
        # x = self.act(x)
        x2 = self.conv2_2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1_1_1x1(x)
        x = self.conv2_1x1(x)
        x = x + y
        x = x + self.mlp(x)
        
        return x

class SK_Block_1(nn.Module):   # num may can set 2
    def __init__(self, in_planes, out_planes, num=2, stride=1, norm_fn='batch', dropout=0,down=False):
        super(SK_Block_1, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(out_planes)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(out_planes)
        elif self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(32, out_planes)
        elif self.norm_fn == 'none':
            self.norm1 = None
        self.conv1 = nn.Conv2d(in_planes,out_planes,1,1,0)
        self.downsample = nn.Sequential(*[
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            
        ])
        self.block = nn.ModuleList()
        [self.block.append(SK_module_1(out_planes, out_planes, stride=stride, norm_fn=norm_fn, dropout=dropout)
                      ) for _ in range(num)]
        
    def forward(self,x):

        x = self.conv1(x)
        for blk in self.block:
            x = blk(x)
        x = self.norm1(x)
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x

class SK_Encoder_1(nn.Module):
    def __init__(self, out_dim=128, stride=1, norm_fn='batch', dropout=0.05,):
        super(SK_Encoder_1, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.in_planes = 32
        self.layer1 = SK_Block_1(32, 64, stride=1)
        self.layer2 = SK_Block_1(64, 96, stride=1)
        self.layer3 = SK_Block_1(96, 128, stride=1)
        # self.layer4 = SK_Block_1(128, 256, stride=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(128, out_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)
            
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
            
        return x

if __name__ == "__main__":

    model = SK_Encoder_1()
    input1 = torch.randn(1, 3, 256, 256)
    input2 = torch.randn(1, 3, 256, 256)
    x = model([input1,input2])
    print(x[0].shape)