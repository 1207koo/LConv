# LConv
Implementation of "LConv: Lineraly-weighted Convolution"

## Environment
Environment below is recommended.
* Python 3.6
* PyTorch 1.9.0


## Use
You can use LConv as torch.nn.Conv2d.

Just include `LConv.py` and try `from LConv import LConv` in your code.

### Parameters
* in_channels
* out_channels
* kernel_size
* stride = 1
* padding = 0
* dilation = 1
* groups = 1
* bias = True
* padding_mode = 'zeros'

### Conversion

```
# convert LConv to torch.nn.Conv2d
lconv = LConv(1, 64, 3)
conv = lconv.convert_to_conv()

# convert torch.nn.Conv2d to LConv
conv = torch.nn.Conv2d(1, 64, 3)
lconv = LConv(1, 64, 3)
lconv.convert_from_conv(conv)
```