import torch.nn as nn
import torch
import torch.nn.functional as F
from data.util import extract_image_patches, reduce_mean, reduce_sum, same_padding, flow_to_image


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def _norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'bn':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'in':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def _activation(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class conv_block(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
                 padding=0, norm='in', activation='relu', pad_type='zero'):
        super(conv_block, self).__init__()
        if pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        if norm == 'in':
            self.norm = nn.InstanceNorm2d(out_nc, affine=False)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(out_nc, affine=True)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported norm type: {}".format(norm)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size, stride, 0, dilation, groups, bias)  # padding=0

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class upconv_block(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, bias=True,
                 padding=0, pad_type='zero', norm='none', activation='relu'):
        super(upconv_block, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_nc, out_nc, 4, 2, 1)
        self.act = _activation('relu')
        self.norm = _norm('in', out_nc)

        self.conv = conv_block(out_nc, out_nc, kernel_size, stride, bias=bias, padding=padding, pad_type=pad_type,
                               norm=norm, activation=activation)

    def forward(self, x):
        x = self.act(self.norm(self.deconv(x)))
        x = self.conv(x)
        return x

class dffm(nn.Module):
    def __init__(self, nc):
        super(dffm, self).__init__()
        self.c1 = conv_layer(nc, nc // 4, 3, 1)
        self.d1 = conv_layer(nc // 4, nc // 4, 3, 1, 1)  # rate = 1
        self.d2 = conv_layer(nc // 4, nc // 4, 3, 1, 2)  # rate = 2
        self.d3 = conv_layer(nc // 4, nc // 4, 3, 1, 4)  # rate = 4
        self.d4 = conv_layer(nc // 4, nc // 4, 3, 1, 8)  # rate = 8
        self.act = _activation('relu')
        self.norm = _norm('in', nc)
        self.c2 = conv_layer(nc, nc, 3, 1)  # fusion

    def forward(self, x):
        output1 = self.act(self.norm(self.c1(x)))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)

        add1 = d1 + d2
        add2 = add1 + d3
        add3 = add2 + d4
        combine = torch.cat([d1, add1, add2, add3], 1)
        output2 = self.c2(self.act(self.norm(combine)))
        output = x + self.norm(output2)
        return output

# Define the resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=64, kernel_size=3, padding=0, dilation=dilation, bias=False),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=False),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        # print(x.shape)
        out = x + self.conv_block(x)
        return out

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class PALayer(nn.Module):
    def __init__(self, input_nc):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(nn.Conv2d(input_nc, input_nc // 8, 1, padding=0, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(input_nc // 8, 1, 1, padding=0, bias=True),
                                nn.Sigmoid())
    def forward(self, x):
        y = self.pa(x)
        y = y * x
        return y

class FAM(nn.Module):
    def __init__(self, input_nc, norm=nn.BatchNorm2d, use_dropout=False, dilation=1):
        super(FAM, self).__init__()

        ###residual block
        res_blocks = []
        for _ in range(4):
            block = ResnetBlock(input_nc, 1)
            res_blocks.append(block)
        self.Res = nn.Sequential(*res_blocks)

        self.cbam = CBAMLayer(input_nc)
        self.pa = PALayer(input_nc)


    def forward(self, x):
        x = self.Res(x)
        x = self.cbam(x)
        x = self.pa(x)
        return x

####partial convolution
class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        # print(input.shape)
        # if mask is not provided, create a mask
        # mask = torch.ones_like(input)
        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
        # print(mask.shape)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes.bool(), 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes.bool(), 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes.bool(), 0.0)
        out = []
        out.append(output)
        out.append(new_mask)
        return out


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='leaky',
                 conv_bias=False, innorm=False, inner=False, outer=False):
        super().__init__()
        if sample == 'same-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == 'same-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.InstanceNorm2d(out_ch, affine=True)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.innorm = innorm
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])

        elif self.innorm:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
        elif self.outer:
            out = self.conv(out)
            out[0] = self.bn(out[0])
        else:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            if hasattr(self, 'activation'):
                out[0] = self.activation(out[0])
        return out


class Pconv(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(Pconv, self).__init__()
        seuqence_3 = []
        seuqence_5 = []
        seuqence_7 = []
        for i in range(5):
            seuqence_3 += [PCBActiv(in_nc, out_nc, innorm=True)]
            seuqence_5 += [PCBActiv(in_nc, out_nc, sample='same-5', innorm=True)]
            seuqence_7 += [PCBActiv(in_nc, out_nc, sample='same-7', innorm=True)]

        self.cov_3 = nn.Sequential(*seuqence_3)
        self.cov_5 = nn.Sequential(*seuqence_5)
        self.cov_7 = nn.Sequential(*seuqence_7)
        self.cov_3 = nn.Sequential(PCBActiv(in_nc, out_nc, innorm=True),
                                   PCBActiv(out_nc, out_nc, innorm=True))
        self.cov_5 = nn.Sequential(PCBActiv(in_nc, out_nc, sample='same-5', innorm=True),
                                   PCBActiv(out_nc, out_nc, sample='same-5', innorm=True))
        self.cov_7 = nn.Sequential(PCBActiv(in_nc, out_nc, sample='same-7', innorm=True),
                                   PCBActiv(out_nc, out_nc, sample='same-7', innorm=True))          
        self.down = nn.Conv2d(in_channels=768, out_channels=256,kernel_size=3,stride=1,padding=1)

    def forward(self, input):
        x = input
        # Multi Scale PConv fill the holes
        x_3 = self.cov_3(x)
        x_5 = self.cov_5(x)
        x_7 = self.cov_7(x)
        x_fuse = torch.cat([x_3[0], x_5[0], x_7[0]], 1)  # 768
        x_out = self.down(x_fuse)


        return x_out

