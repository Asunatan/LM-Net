import torch.nn as nn


class involution(nn.Module):

    def __init__(self,
                 in_channels,
                 kernel_size,
                 stride):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        reduction_ratio = 4
        self.group_channels = 4
        self.groups = self.in_channels // self.group_channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels,in_channels// reduction_ratio,1),
                                   nn.BatchNorm2d(in_channels// reduction_ratio),
                                   nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels// reduction_ratio,kernel_size**2 * self.groups,1)

        )
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.in_channels, h, w)
        return out
