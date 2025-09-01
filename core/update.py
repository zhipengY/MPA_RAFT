import torch
import torch.nn as nn
import torch.nn.functional as F
from gma import Aggregate
from extractor import MPA

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
        # self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        # self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        # self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

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

class ConvGRU_gma(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256 + 128):
        super(ConvGRU_gma, self).__init__()
        # self.conv1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 5, padding=2)
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
        # self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        # self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        # self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        # self.conv = nn.Conv2d(128, 80, 3, padding=1)
        # self.MPA = MPA()

        self.convc1 = nn.Conv2d(cor_planes, 128, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 96, 3, padding=1)
        self.conv = nn.Conv2d(128+96, 128-2, 3, padding=1)

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
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=128 + 64)
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
            nn.Conv2d(128, 128, 3, padding=1),
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


class GMAMotionEncoder(nn.Module):
    def __init__(self, args):
        super(GMAMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 128, 1, padding=0)
        self.convc2 = nn.Conv2d(128,128,3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 96, 3, padding=1)
        self.conv = nn.Conv2d(128+96, 128-2, 3, padding=1)
        self.MPA = MPA(128)


    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        cor = self.MPA(cor)
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class GMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(GMAUpdateBlock, self).__init__()
        self.encoder = GMAMotionEncoder(args)
        self.gru = ConvGRU_gma(hidden_dim=hidden_dim, input_dim=128+64+128)  # GMA 128 + 64 + 128 no GMA 128 + 64
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)
        self.aggregator = Aggregate(args=args, dim=128, dim_head=128, heads=8)
        self.pool1 = nn.MaxPool2d(3, 1)
        self.mask = nn.Sequential(
            nn.Conv2d(96, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 64*9, 1, padding=0))
            nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow, Attention):
    # def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        motion_aggregate = self.aggregator(Attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_aggregate], dim=1)
        # inp_cat = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp_cat)
        delta_flow = self.flow_head(net)
        mask = .25 * self.mask(net)

        return net, mask, delta_flow
        # return net, None, delta_flow
    
class GMAUpdateBlock_quarter(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(GMAUpdateBlock_quarter, self).__init__()
        self.encoder = GMAMotionEncoder(args)
        self.gru = ConvGRU_gma(hidden_dim=hidden_dim, input_dim=128+64)  # GMA 128 + 64 + 128 no GMA 128 + 64
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)
        self.aggregator = Aggregate(args=args, dim=128, dim_head=128, heads=8)
        # self.pool1 = nn.MaxPool2d(3, 1)
        self.mask = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 16*9, 1, padding=0))

    # def forward(self, net, inp, corr, flow, Attention):
    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        # motion_aggregate = self.aggregator(Attention, motion_features)
        # inp_cat = torch.cat([inp, motion_features, motion_aggregate], dim=1)
        inp_cat = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp_cat)
        delta_flow = self.flow_head(net)
        mask = .25 * self.mask(net)

        return net, mask, delta_flow


