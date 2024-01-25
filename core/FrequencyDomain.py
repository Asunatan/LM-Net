import torch.nn as nn
import torch
from resnet import resnet50
#from torchvision.models import resnet50
class TwinTowers(nn.Module):
    def __init__(self, cnn,n_classes=2, filters=[16,32, 64, 128, 256],deep_supervision=False):
        super(TwinTowers, self).__init__()
        self.cnn=resnet50()






    def forward(self, x,DCT):
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)




        a=0

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    y = torch.randn(4, 192, 32, 32)
    # detail = Detail_Branch()
    # feat = detail(x)
    # print('detail', feat.size())

    net = TwinTowers(resnet50())
    logits = net(x)