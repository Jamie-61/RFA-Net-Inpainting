import torch
import torch.nn as nn
from . import block as B
import torchvision


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


class RFA(nn.Module):
    def __init__(self,  blocks, conv=default_conv, norm='in', activation='relu'):
        super(FFA, self).__init__()
        # self.gps = gps
        self.nf = 64
        kernel_size = 3
        pre_process = [conv(4, self.nf, kernel_size)]
        # assert self.gps == 4
        self.g1 = Group(conv, self.nf, kernel_size, blocks=blocks)  # [64,256,256]
        self.g2 = Group(conv, self.nf, kernel_size, blocks=blocks)  # [64,256,256]
        self.g3 = Group(conv, self.nf, kernel_size, blocks=blocks)
        self.g4 = Group(conv, self.nf, kernel_size, blocks=blocks)

        self.pre = nn.Sequential(*pre_process)
  

    def forward(self, x1):

        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        res4 = self.g4(res3)
        x = torch.cat([res1, res2, res3, res4], dim=1)
        return x

class InpaintingGenerator(nn.Module):
    def __init__(self, in_nc, out_nc, nf, n_res, norm='in', activation='relu'):
        super(InpaintingGenerator, self).__init__()

        self.conv1 = nn.Sequential(B.conv_block(4 * nf, 4 * nf, 3, stride=2, padding=1, norm=norm, activation=activation),
                                  B.conv_block(4 * nf, 4 * nf, 3, stride=2, padding=1, norm=norm, activation=activation))

        self.rfa = RFA(1)
        blocks = []
        for _ in range(n_res):
            block = B.dffm(4 * nf)
            blocks.append(block)

        self.dffm = nn.Sequential(*blocks)

        self.pconv = B.Pconv(4 * nf, 4 * nf)

        self.decoder = nn.Sequential(
            B.conv_block(4 * nf, 4 * nf, 3, stride=1, padding=1, norm=norm, activation=activation),  # [256, 64, 64]
            B.upconv_block(4 * nf, 2 * nf, kernel_size=3, stride=1, padding=1, norm=norm, activation='relu'),
            # [128, 128, 128]
            B.upconv_block(2 * nf, nf, kernel_size=3, stride=1, padding=1, norm=norm, activation='relu'),
            # [64, 256, 256]
            B.conv_block(nf, out_nc, 3, stride=1, padding=1, norm='none', activation='tanh')  # [3, 256, 256]
        )
        self.conv2 = nn.Sequential(B.conv_block(4 * nf, 2 * nf, kernel_size=3, stride=1, padding=1, norm=norm, activation='relu'),
                                  B.conv_block(2 * nf, nf, kernel_size=3, stride=1, padding=1, norm=norm, activation='relu'),
                                  B.conv_block(nf, out_nc, 3, stride=1, padding=1, norm='none', activation='tanh'))


    def forward(self, y):
        x = self.rfa(y)
        x = self.conv1(x)
        x = self.pconv(x)
        x = self.dffm(x)
        x = self.conv2(x)
        x = y + x
        return x


class Discriminator(nn.Module):
    def __init__(self, in_nc, nf, norm='bn', activation='lrelu'):
        super(Discriminator, self).__init__()
        global_model = []
        global_model += [B.conv_block(in_nc, nf, 5, stride=2, padding=2, norm=norm, activation=activation),
                         # [64, 128, 128]
                         B.conv_block(nf, 2 * nf, 5, stride=2, padding=2, norm=norm, activation=activation),
                         # [128, 64, 64]
                         B.conv_block(2 * nf, 4 * nf, 5, stride=2, padding=2, norm=norm, activation=activation),
                         # [256, 32, 32]
                         B.conv_block(4 * nf, 8 * nf, 5, stride=2, padding=2, norm=norm, activation=activation),
                         # [512, 16, 16]
                         B.conv_block(8 * nf, 8 * nf, 5, stride=2, padding=2, norm=norm, activation=activation),
                         # [512, 8, 8]
                         B.conv_block(8 * nf, 8 * nf, 5, stride=2, padding=2, norm=norm,
                                      activation=activation)]  # [512, 4, 4]
        self.global_model = nn.Sequential(*global_model)


        self.local_fea1 = B.conv_block(in_nc, nf, 5, stride=2, padding=2, norm=norm, activation=activation)  # [64, 64, 64]
        self.local_fea2 = B.conv_block(nf, 2 * nf, 5, stride=2, padding=2, norm=norm, activation=activation)  # [128, 32, 32]
        self.local_fea3 = B.conv_block(2 * nf, 4 * nf, 5, stride=2, padding=2, norm=norm, activation=activation)  # [256, 16, 16]
        self.local_fea4 = B.conv_block(4 * nf, 8 * nf, 5, stride=2, padding=2, norm=norm, activation=activation)  # [512, 8, 8]
        self.local_fea5 = B.conv_block(8 * nf, 8 * nf, 5, stride=2, padding=2, norm=norm, activation=activation)  # [512, 4, 4]

        self.global_classifier = nn.Linear(512 * 4 * 4, 512)
        self.local_classifier = nn.Linear(512 * 4 * 4, 512)
        self.classifier = nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(1024, 1))

    def forward(self, x_local, x_global):
        out_local_fea1 = self.local_fea1(x_local)
        out_local_fea2 = self.local_fea2(out_local_fea1)
        out_local_fea3 = self.local_fea3(out_local_fea2)
        out_local_fea4 = self.local_fea4(out_local_fea3)
        out_local_fea5 = self.local_fea5(out_local_fea4)
        out_local = out_local_fea5.view(out_local_fea5.size(0), -1)
        out_local = self.local_classifier(out_local)

        out_global = self.global_model(x_global)
        out_global = out_global.view(out_global.size(0), -1)
        out_global = self.global_classifier(out_global)

        out = torch.cat([out_local, out_global], dim=1)
        out = self.classifier(out)
        return out, [out_local_fea1, out_local_fea2, out_local_fea3, out_local_fea4, out_local_fea5]


# Feature extractor
# data range [-1, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=(1, 6, 11, 20, 29), use_input_norm=True):
        super(VGGFeatureExtractor, self).__init__()
        model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485 - 1, 0.456 - 1, 0.406 - 1]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229 * 2, 0.224 * 2, 0.225 * 2]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        for k, v in model.named_parameters():
            v.requires_grad = False

        self.relu1_1 = nn.Sequential(*list(model.features.children())[:(feature_layer[0] + 1)])  # [0-1]
        self.relu2_1 = nn.Sequential(
            *list(model.features.children())[(feature_layer[0] + 1):(feature_layer[1] + 1)])  # [2-6]
        self.relu3_1 = nn.Sequential(
            *list(model.features.children())[(feature_layer[1] + 1):(feature_layer[2] + 1)])  # [7-11]
        self.relu4_1 = nn.Sequential(
            *list(model.features.children())[(feature_layer[2] + 1):(feature_layer[3] + 1)])  # [12-20]
        self.relu5_1 = nn.Sequential(
            *list(model.features.children())[(feature_layer[3] + 1):(feature_layer[4] + 1)])  # [21-29]

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        relu1_1 = self.relu1_1(x)
        relu2_1 = self.relu2_1(relu1_1)
        relu3_1 = self.relu3_1(relu2_1)
        relu4_1 = self.relu4_1(relu3_1)
        relu5_1 = self.relu5_1(relu4_1)
        return [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]