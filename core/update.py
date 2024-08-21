import torch
import torch.nn as nn
import torch.nn.functional as F
from opts import cross_attention, SpitalCrossAttention
from gma import Aggregate
from core.extractor import ESCA


class ESCA1(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ESCA1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dim = channel
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2)
        self.conv2_1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv2_2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv2_3 = nn.Conv2d(2 , channel, kernel_size=1)
        self.conv3_1 = nn.Conv2d(2 * channel, channel, kernel_size=1)
        self.conv3_2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv_h = nn.Conv2d(1, 1, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.conv_w = nn.Conv2d(1, 1, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.convout = nn.Conv2d(2 * channel, channel, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(channel)
        self.lrelu = nn.LeakyReLU()
        self.maxpool1 = nn.Conv2d(channel,1,1)

    def forward(self, x):

        b, c, h, w = x.size()
        x1 = self.maxpool1(x)

        y = self.avg_pool(x)
        # x_m = self.norm(x1)

        x_h = self.conv_h(x1)
        x_w = self.conv_w(x1)
        # x_1 = self.conv2_2(x_m)

        x_c = torch.cat((x_h, x_w), dim=1)
        x_c = self.conv2_3(x_c)
        x_c = self.norm(x_c)
        x_c = self.lrelu(x_c)

        # x_2 = torch.cat((x_c, x), dim=1)
        # x_2 = self.conv3_1(x_2)
        # x_2 = self.conv3_2(x_2)

        out = self.sigmoid(x_c)

        y = self.conv1(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        x = torch.cat((x * y.expand_as(x), out * x), dim=1)
        x = self.convout(x)

        return x
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h

class ConvGRU_gma(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256 + 128):
        super(ConvGRU_gma, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 5, padding=2)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 5, padding=2)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 5, padding=2)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)
        # self.esca = ESCA()

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + 64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow


class Cross_update_flow(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(Cross_update_flow, self).__init__()

        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.cross_a = SpitalCrossAttention(in_channel=196)
        self.flow_head = FlowHead(input_dim=196, hidden_dim=256)
        self.combine = nn.Sequential(
            nn.Conv2d(288, 256, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 196, 1, padding=0))
        self.conv1 = nn.Conv2d(198, 160, 1)

    def forward(self, net, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([net, motion_features], dim=1)
        inp = self.combine(inp)

        net = self.cross_a(corr, inp)
        flow = self.flow_head(net)
        net = self.conv1(torch.cat([net, flow], dim=1))

        # scale mask to balence gradients
        # mask = .25 * self.mask(net)
        return net, flow


class GMAMotionEncoder(nn.Module):
    def __init__(self, args):
        super(GMAMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 128, 1, padding=0)
        self.convc2 = nn.Conv2d(128,128,3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 96, 3, padding=1)
        self.conv = nn.Conv2d(128+96, 128-2, 3, padding=1)
        self.esca = ESCA(128)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        cor = self.esca(cor)
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class hwa(nn.Module):
    def __init__(self, in_channels):
        super(hwa, self).__init__()
        self.pool1 = nn.MaxPool2d((3, 1), 1, (3, 1))
        self.pool2 = nn.MaxPool2d((1, 3), 1, (1, 3))
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, 1, 1)
        self.norm = nn.GroupNorm(16, in_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.norm(x)
        y1 = self.pool1(y)
        y2 = self.pool2(y)
        y_ = torch.cat((y1, y2), dim=1)
        y_ = self.conv1(y_)
        out = y_ * x
        out = self.gelu(out)
        return out

class GMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(GMAUpdateBlock, self).__init__()
        self.encoder = GMAMotionEncoder(args)
        self.gru = ConvGRU_gma(hidden_dim=hidden_dim, input_dim=128+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)
        self.aggregator = Aggregate(args=args, dim=128, dim_head=128, heads=8)
        self.pool1 = nn.MaxPool2d(3, 1)
        self.mask = nn.Sequential(
            nn.Conv2d(96, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, Attention):
    # def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        motion_aggregate = self.aggregator(Attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_aggregate], dim=1)
        inp_cat = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp_cat)
        delta_flow = self.flow_head(net)
        mask = .25 * self.mask(net)

        return net, mask, delta_flow
