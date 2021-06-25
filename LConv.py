import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size if type(kernel_size) in [tuple, list] and len(kernel_size) == 2 else (kernel_size, kernel_size)
    self.stride = stride if type(stride) in [tuple, list] and len(stride) == 2 else (stride, stride)
    self.padding = padding if type(padding) in [tuple, list] and len(padding) == 2 else (padding, padding)
    self.dilation = dilation if type(dilation) in [tuple, list] and len(dilation) == 2 else (dilation, dilation)
    self.groups = groups
    self.padding_mode = 'constant' if padding_mode=='zeros' else padding_mode
    abc = torch.Tensor(3 * in_channels // groups, out_channels)
    self.abc = nn.Parameter(abc)
    if bias:
      bias_ = torch.Tensor(out_channels)
      self.bias = nn.Parameter(bias_)
    else:
      self.bias = None
    self.reset_parameters()
  
  def reset_parameters(self):
    nn.init.kaiming_uniform_(self.abc, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.abc)
      bound = 1 / math.sqrt(fan_in)
      nn.init.uniform_(self.bias, -bound, bound)

  def convert_to_conv(self):
    device = self.abc.device
    conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, (self.bias is not None), self.padding_mode if self.padding_mode!='constant' else 'zeros')
    a, b, c = self.abc.view(3, self.in_channels // self.groups, self.out_channels, 1, 1).transpose(1, 2).repeat(1, 1, 1, self.kernel_size[0], self.kernel_size[1])
    xi = torch.arange(self.kernel_size[0], device=device).float().sub_((self.kernel_size[0]-1)/2.0).view(self.kernel_size[0], 1).repeat(1, self.kernel_size[1])
    yi = torch.arange(self.kernel_size[1], device=device).float().sub_((self.kernel_size[1]-1)/2.0).view(1, self.kernel_size[1]).repeat(self.kernel_size[0], 1)
    conv.weight = nn.Parameter(a * xi + b * yi + c)
    conv.bias = self.bias
    return conv
  
  def convert_from_conv(self, conv):
    device = self.abc.device
    w = conv.weight.transpose(0, 1).to(device)
    xi = torch.arange(self.kernel_size[0], device=device).float().sub_((self.kernel_size[0]-1)/2.0).view(self.kernel_size[0], 1).repeat(1, self.kernel_size[1])
    yi = torch.arange(self.kernel_size[1], device=device).float().sub_((self.kernel_size[1]-1)/2.0).view(1, self.kernel_size[1]).repeat(self.kernel_size[0], 1)
    oi = torch.ones_like(xi, device=device)
    a = (xi * w).sum((2, 3)) / ((xi * xi).sum() + 1e-9)
    b = (yi * w).sum((2, 3)) / ((yi * yi).sum() + 1e-9)
    c = (oi * w).sum((2, 3)) / ((oi * oi).sum() + 1e-9)
    self.abc = nn.Parameter(torch.cat((a, b, c), 0))
    if conv.bias is None:
      self.bias = None
    else:
      self.bias = conv.bias.to(device)

  def forward(self, data):
    k0, k1 = self.kernel_size
    s0, s1 = self.stride
    p0, p1 = self.padding
    d0, d1 = self.dilation
    g = self.groups
    dk0, dk1 = k0 * d0, k1 * d1
    device = data.device
    ph, pw = data.size(2) + 2 * p0 + d0, data.size(3) + 2 * p1 + d1
    data_p = F.pad(F.pad(data, (p1, p1, p0, p0), self.padding_mode), (d1, (d1 - pw % d1) % d1, d0, (d0 - ph % d0) % d0))
    b, c, h, w = data_p.size()
    xi = torch.arange(h, device=device).float().div_(d0).view(h, 1).repeat(1, w)
    yi = torch.arange(w, device=device).float().div_(d1).view(1, w).repeat(h, 1)
    data_c = torch.cat((data_p * xi, data_p * yi, data_p), 1)
    data_c = data_c.view(b, 3 * c, h // d0, d0, w).cumsum_(2).view(b, 3 * c, h, w)[:, :, :ph]
    data_c = data_c[:, :, dk0:ph:s0] - data_c[:, :, 0:ph-dk0:s0]
    data_c = data_c.view(b, 3 * c, -1, w // d1, d1).cumsum_(3).view(b, 3 * c, -1, w)[:, :, :, :pw]
    data_c = data_c[:, :, :, dk1:pw:s1] - data_c[:, :, :, 0:pw-dk1:s1]

    data_c[:, 0:c].sub_(data_c[:, 2*c:] * (xi[0:ph-dk0:s0, 0:pw-dk1:s1] + (k0+1)/2))
    data_c[:, c:2*c].sub_(data_c[:, 2*c:] * (yi[0:ph-dk0:s0, 0:pw-dk1:s1] + (k1+1)/2))
    _, _, h, w = data_c.size()
    data_c = data_c.view(b, 3, g, c // g, h, w)
    data_c = torch.cat((data_c[:, 0], data_c[:, 1], data_c[:, 2]), 2).view(b, 3 * c, h, w)
    data_c.transpose_(1, 3)

    output_list = []
    ni, no = 3 * self.in_channels // g, self.out_channels // g
    for i in range(g):
      output_list.append(torch.matmul(data_c[:, :, :, i*ni:(i+1)*ni], self.abc[:, i*no:(i+1)*no]))
    output = torch.cat(output_list, 3)
    return (output.transpose(1, 3) if self.bias is None else output.add_(self.bias).transpose(1, 3))