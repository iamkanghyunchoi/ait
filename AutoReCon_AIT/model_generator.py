import torch
from torch import nn
import torch.nn.functional as F
from conditional_batchnorm import CategoricalConditionalBatchNorm2d

# 7
PRIMITIVES_op = [
  'none',
  'skip_connect',
  'conv_1x1',
  'conv_3x3',
  'conv_5x5',
  'dil_conv_3x3',
  'dil_conv_5x5'
]

# 2
PRIMITIVES_up = [
  'nearest',
  'bilinear',
]

# 3
PRIMITIVES_cross = [
  'none_up',
  'nearest',
  'bilinear',
]

# ------------------------------------------------------------------------------------------------------------------- #
OPS = {
    'none': lambda in_ch, out_ch, stride, sn, act: Zero(),
    'skip_connect': lambda in_ch, out_ch, stride, sn, act: Identity(),
    'conv_1x1': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 1, stride, 0, sn, act),
    'conv_3x3': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 3, stride, 1, sn, act),
    'conv_5x5': lambda in_ch, out_ch, stride, sn, act: Conv(in_ch, out_ch, 5, stride, 2, sn, act),
    'dil_conv_3x3': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 3, stride, 2, 2, sn, act),
    'dil_conv_5x5': lambda in_ch, out_ch, stride, sn, act: DilConv(in_ch, out_ch, 5, stride, 4, 2, sn, act)
}

UPS = {
    'nearest': lambda in_ch, out_ch: Up(in_ch, out_ch, mode='nearest'),
    'bilinear': lambda in_ch, out_ch: Up(in_ch, out_ch, mode='bilinear')
#    'ConvTranspose': lambda in_ch, out_ch: Up(in_ch, out_ch, mode='convT')
}

UPS_cross = {
    'none_up': lambda in_ch, out_ch: Up_none(in_ch, out_ch, mode='nearest'),
    'nearest': lambda in_ch, out_ch: Up(in_ch, out_ch, mode='nearest'),
    'bilinear': lambda in_ch, out_ch: Up(in_ch, out_ch, mode='bilinear')
}


# ------------------------------------------------------------------------------------------------------------------- #

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, sn, act):
        super(Conv, self).__init__()
        if sn:
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding))
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        if act:
            self.conditional_bn = CategoricalConditionalBatchNorm2d(1000, in_ch, 0.8)
            self.op = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), self.conv)
        else:
            self.op = nn.Sequential(self.conv)
            
    def forward(self, x, labels):
        output = self.conditional_bn(x, labels)
        output = self.op(output)
        return output

class DilConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, sn, act):
        super(DilConv, self).__init__()
        if sn:
            self.dilconv = nn.utils.spectral_norm(
              nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
        else:
            self.dilconv = \
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        if act:
            self.conditional_bn = CategoricalConditionalBatchNorm2d(1000, in_ch, 0.8)
            self.op = nn.Sequential( nn.LeakyReLU(0.2, inplace=True), self.dilconv)
        else:
            self.op = nn.Sequential(self.dilconv)

    def forward(self, x, labels):
        output = self.conditional_bn(x, labels)
        output = self.op(output)
        return output


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
  
    def forward(self, x, labels):
        return x

class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()
  
    def forward(self, x, labels):
        return x.mul(0.)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, mode=None):
        super(Up, self).__init__()
        self.up_mode = mode
        if self.up_mode == 'convT':
            self.convT = nn.Sequential(
                CategoricalConditionalBatchNorm2d(1000, in_ch, 0.8), 
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(
                    in_ch, in_ch, kernel_size=3, stride=2, padding=1, output_padding=1, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)
            )
        else:
            self.conditional_bn = CategoricalConditionalBatchNorm2d(1000, in_ch, 0.8)
            self.c = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1)
            )
    def forward(self, x, labels):
        if self.up_mode == 'convT':
            return self.convT(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.up_mode)
            output = self.conditional_bn(x, labels)
            output = self.c(output)
            return output
class Up_none(nn.Module):
    def __init__(self, in_ch, out_ch, mode=None):
        super(Up_none, self).__init__()
        self.up_mode = mode
        if self.up_mode == 'convT':
            self.convT = nn.Sequential(
                CategoricalConditionalBatchNorm2d(1000, in_ch, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(
                    in_ch, in_ch, kernel_size=3, stride=2, padding=1, output_padding=1, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)
            )
        else:
            self.conditional_bn = CategoricalConditionalBatchNorm2d(1000, in_ch, 0.8)
            self.c = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1)
            )

    def forward(self, x, labels):
        if self.up_mode == 'convT':
            return self.convT(x).mul(0.)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.up_mode)
            output = self.conditional_bn(x, labels)
            output = self.c(output).mul(0.0)
            return output


class MixedOp(nn.Module):
    def __init__(self, in_ch, out_ch, stride, sn, act, primitives, vector_op):
        super(MixedOp, self).__init__()
        self.ops = nn.ModuleList()
        self.vector_op = vector_op
        count = 0
        for primitive in primitives:
            if ( self.vector_op[count] == torch.tensor([1.0], requires_grad=False) ):
                op = OPS[primitive](in_ch, out_ch, stride, sn, act)
                self.ops.append(op)
            count = count + 1

    def forward(self, x, labels):
        output = self.ops[0](x, labels)
        return output
 
class MixedUp(nn.Module):
    def __init__(self, in_ch, out_ch, primitives, vector_up):
        super(MixedUp, self).__init__()
        self.ups = nn.ModuleList()
        self.vector_up = vector_up
        count = 0
        for primitive in primitives:
            if ( self.vector_up[count] == torch.tensor([1.0], requires_grad=False) ):
                up = UPS[primitive](in_ch, out_ch)
                self.ups.append(up)
            count = count + 1

    def forward(self, x, labels):
        output = self.ups[0](x, labels)
        return output

    
class MixedUp_cross(nn.Module):
    def __init__(self, in_ch, out_ch, primitives, vector_cross):
        super(MixedUp_cross, self).__init__()
        self.ups_cross = nn.ModuleList()
        self.vector_cross = vector_cross
        count = 0
        for primitive in primitives:
            if ( self.vector_cross[count] == torch.tensor([1.0], requires_grad=False) ):
                up_cross = UPS_cross[primitive](in_ch, out_ch)
                self.ups_cross.append(up_cross)

            count = count + 1
    
    def forward(self, x, labels):
        output = self.ups_cross[0](x, labels)
        return output

# ------------------------------------------------------------------------------------------------------------------- #

class Cell_zero(nn.Module):
    def __init__(self, in_channels, out_channels, matrix_up, matrix_op):
        super(Cell_zero, self).__init__()

        self.up0 = MixedUp(in_channels, out_channels, primitives=PRIMITIVES_up, vector_up=matrix_up[0])
        self.up1 = MixedUp(in_channels, out_channels, primitives=PRIMITIVES_up, vector_up=matrix_up[1])

        self.c0 = MixedOp(out_channels, out_channels, 1, False, True, primitives=PRIMITIVES_op, vector_op=matrix_op[0])
        self.c1 = MixedOp(out_channels, out_channels, 1, False, True, primitives=PRIMITIVES_op, vector_op=matrix_op[1])
        self.c2 = MixedOp(out_channels, out_channels, 1, False, True, primitives=PRIMITIVES_op, vector_op=matrix_op[2])
        self.c3 = MixedOp(out_channels, out_channels, 1, False, True, primitives=PRIMITIVES_op, vector_op=matrix_op[3])
        self.c4 = MixedOp(out_channels, out_channels, 1, False, True, primitives=PRIMITIVES_op, vector_op=matrix_op[4])

    def forward(self, x, labels):

        node0 = self.up0(x, labels)
        node1 = self.up1(x, labels)

        node2 = self.c0(node0, labels) + self.c1(node1, labels)
        node3 = self.c2(node0, labels) + self.c3(node1, labels) + self.c4(node2, labels)
        return node3, node0, node1, node2

class Cell_one(nn.Module):
    def __init__(self, in_channels, out_channels, matrix_up, matrix_op, matrix_cross):
        super(Cell_one, self).__init__()

        self.up0 = MixedUp(in_channels, out_channels, primitives=PRIMITIVES_up, vector_up=matrix_up[0])
        self.up1 = MixedUp(in_channels, out_channels, primitives=PRIMITIVES_up, vector_up=matrix_up[1])

        self.c0 = MixedOp(out_channels, out_channels, 1, False, True, primitives=PRIMITIVES_op, vector_op=matrix_op[0])
        self.c1 = MixedOp(out_channels, out_channels, 1, False, True, primitives=PRIMITIVES_op, vector_op=matrix_op[1])
        self.c2 = MixedOp(out_channels, out_channels, 1, False, True, primitives=PRIMITIVES_op, vector_op=matrix_op[2])
        self.c3 = MixedOp(out_channels, out_channels, 1, False, True, primitives=PRIMITIVES_op, vector_op=matrix_op[3])
        self.c4 = MixedOp(out_channels, out_channels, 1, False, True, primitives=PRIMITIVES_op, vector_op=matrix_op[4])

        self.up0_c01 = MixedUp_cross(in_channels, out_channels, primitives=PRIMITIVES_cross, vector_cross=matrix_cross[0])
        self.up1_c01 = MixedUp_cross(in_channels, out_channels, primitives=PRIMITIVES_cross, vector_cross=matrix_cross[1])
        self.up2_c01 = MixedUp_cross(in_channels, out_channels, primitives=PRIMITIVES_cross, vector_cross=matrix_cross[2])
        self.up3_c01 = MixedUp_cross(in_channels, out_channels, primitives=PRIMITIVES_cross, vector_cross=matrix_cross[3])
        self.up4_c01 = MixedUp_cross(in_channels, out_channels, primitives=PRIMITIVES_cross, vector_cross=matrix_cross[4])
        self.up5_c01 = MixedUp_cross(in_channels, out_channels, primitives=PRIMITIVES_cross, vector_cross=matrix_cross[5])

    def forward(self, node3_c0, node0_c0, node1_c0, node2_c0, labels):

        node0 = self.up0(node3_c0, labels)
        node1 = self.up1(node3_c0, labels)
 
        node2 = self.c0(node0, labels) + self.c1(node1, labels)
        node2 = self.up0_c01(node0_c0, labels) + self.up1_c01(node1_c0, labels) + self.up2_c01(node2_c0, labels) + node2

        node3 = self.c2(node0, labels) + self.c3(node1, labels) + self.c4(node2, labels)
        node3 = self.up3_c01(node0_c0, labels) + self.up4_c01(node1_c0, labels) + self.up5_c01(node2_c0, labels) + node3
        return node3, node0, node1, node2

class Generator(nn.Module):
    def __init__(self, arch_param, options):
        super(Generator, self).__init__()

        self.settings = options

        self.gf_dim = 256
        self.nclass = self.settings.nClasses
        if self.settings.dataset in ["cifar10","cifar100"]:
            self.bottom_width = 32 // 8
        else:
            self.bottom_width = 224 // 8
        self.latent_dim = self.settings.latent_dim
        self.base0_latent_dim = self.latent_dim // 3 
        self.base1_latent_dim = self.latent_dim // 3 
        self.base2_latent_dim = self.latent_dim - self.base0_latent_dim*2 

        self.num_up_operations = 2
        self.num_op_operations = 7
        self.num_cross_operations = 3
        self.num_rows_up = 6
        self.num_rows_op = 15
        self.num_rows_cross = 12

        self.thetas_matrix_up, self.thetas_matrix_op, self.thetas_matrix_cross = arch_param

        self.l1 = nn.Linear(self.base0_latent_dim, (self.bottom_width ** 2) * (self.gf_dim // 4))
        self.l2 = nn.Linear(self.base1_latent_dim, ((self.bottom_width * 2) ** 2) * (self.gf_dim // 4))
        self.l3 = nn.Linear(self.base2_latent_dim, ((self.bottom_width * 4) ** 2) * (self.gf_dim // 4))

        self.cell0 = Cell_zero(self.gf_dim // 4, self.gf_dim // 4, self.thetas_matrix_up[0:2].clone(), self.thetas_matrix_op[0:5].clone())
        self.cell1 = Cell_one(self.gf_dim // 4, self.gf_dim // 4, self.thetas_matrix_up[2:4].clone(), self.thetas_matrix_op[5:10].clone(), self.thetas_matrix_cross[0:6].clone())
        self.cell2 = Cell_one(self.gf_dim // 4, self.gf_dim // 4, self.thetas_matrix_up[4:6].clone(), self.thetas_matrix_op[10:15].clone(), self.thetas_matrix_cross[6:12].clone())
        self.to_rgb_conditional_bn = CategoricalConditionalBatchNorm2d(self.nclass, self.gf_dim // 4, 0.8)
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.gf_dim // 4, 3, 3, 1, 1),
            nn.Tanh(),
            nn.BatchNorm2d(3, affine=False)
        )

    def forward(self, z, labels):

        gen_input = z
        node_in0 = self.l1(gen_input[:, :self.base0_latent_dim]).view(-1, self.gf_dim // 4, self.bottom_width, self.bottom_width)
        node_in1 = self.l2(gen_input[:, self.base0_latent_dim:(self.base0_latent_dim + self.base1_latent_dim)]).view(-1, self.gf_dim // 4, self.bottom_width * 2, self.bottom_width * 2)
        node_in2 = self.l3(gen_input[:, (self.base0_latent_dim + self.base1_latent_dim):]).view(-1, self.gf_dim // 4, self.bottom_width * 4, self.bottom_width * 4)

        node3_c0, node0_c0, node1_c0, node2_c0 = self.cell0(node_in0, labels)
        node3_c1, node0_c1, node1_c1, node2_c1 = self.cell1(node3_c0 + node_in1, node0_c0, node1_c0, node2_c0, labels)
        node3_c2, node0_c2, node1_c2, node2_c2 = self.cell2(node3_c1 + node_in2, node0_c1, node1_c1, node2_c1, labels)
        node_out = self.to_rgb_conditional_bn(node3_c2, labels)
        node_out = self.to_rgb(node_out)
        return node_out


