import torch.nn as nn
import torch
from .modules import *
#from .nonlocal_block import NONLocalBlock2D
from .blur_pool import BlurPool2d
#from SoftPool import SoftPool2d
#from .nattencuda import NeighborhoodAttention,NEWNeighborhoodAttention
#from .nattentorch import LegacyNeighborhoodAttention
#from adaPool import AdaPool2d
#from .involution_cuda import involution
class MyUnet(nn.Module):
    def __init__(self, channel,n_classes=2, filters=[16,32, 64, 128, 256],deep_supervision=False):
        super(MyUnet, self).__init__()
        self.deep_supervision=deep_supervision
        self.filters=filters

        self.conv1=nn.Sequential(ReparamConv(channel,filters[1],filters[0],5,3),
                                 ReparamConv(filters[0], filters[1], filters[0],5,3),
                                 )
        #self.eca1=ECA(filters[0])
        self.down1=nn.Sequential(
            #nn.AvgPool2d(3,2,1),
            nn.Conv2d(filters[0],filters[1],3,2,1),
            nn.BatchNorm2d(filters[1]),
            nn.GELU()
            #nn.MaxPool2d(3,2,1),
            #BlurPool2d(filters[0]),
            #SoftPool2d(2, 2,True),
            #nn.Conv2d(filters[0],filters[1],1)
        )

        self.conv2=nn.Sequential(ReparamConv(filters[1],filters[2],filters[1],5,3),
                                 ReparamConv(filters[1], filters[2], filters[1],5,3),
                                 )
        #self.eca2 = ECA(filters[1])
        self.down2=nn.Sequential(
            #nn.AvgPool2d(3,2,1),
            #nn.Conv2d(filters[1],filters[1],3,2,1),
            #nn.MaxPool2d(3,2,1),
            #BlurPool2d(filters[1]),
            #SoftPool2d(2, 2,True),
            #nn.Conv2d(filters[1], filters[2], 1)
            nn.Conv2d(filters[1], filters[2], 3, 2,1),
            nn.BatchNorm2d(filters[2]),
            nn.GELU()
             )

        self.conv3=nn.Sequential(ReparamConv(filters[2],filters[3],filters[2],5,3),
                                 ReparamConv(filters[2], filters[3], filters[2],5,3),

                                 )
        #self.eca3 = ECA(filters[2])
        self.down3=nn.Sequential(
            #nn.AvgPool2d(3,2,1),
            nn.Conv2d(filters[2],filters[3],3,2,1),
            nn.BatchNorm2d(filters[3]),
            nn.GELU()
            #nn.MaxPool2d(3,2,1),
            #BlurPool2d(filters[2]),
            #SoftPool2d(2, 2,True),
            #nn.Conv2d(filters[2], filters[3], 1)
        )

        self.conv4=nn.Sequential(ReparamConv(filters[3],filters[4],filters[3],5,3),
                                 ReparamConv(filters[3], filters[4], filters[3],5,3),

                                 )
        #self.eca4 = ECA(filters[3])
        self.down4=nn.Sequential(
            #nn.AvgPool2d(3,2,1),
            nn.Conv2d(filters[3],filters[4],3,2,1),
            nn.BatchNorm2d(filters[4]),
            nn.GELU()
            #nn.MaxPool2d(3,2,1),
            #BlurPool2d(filters[3]),
            #SoftPool2d(2, 2,True),
            #nn.Conv2d(filters[3], filters[4], 1)
        )
        self.dconv1=nn.Sequential(ReparamConv(filters[3],filters[4],filters[3],5,3),
                                 ReparamConv(filters[3], filters[4], filters[3],5,3),

                                 )
        self.dconv2=nn.Sequential(ReparamConv(filters[2],filters[3],filters[2],5,3),
                                 ReparamConv(filters[2], filters[3], filters[2],5,3),

                                 )
        self.dconv3=nn.Sequential(ReparamConv(filters[1],filters[2],filters[1],5,3),
                                 ReparamConv(filters[1], filters[2], filters[1],5,3),
                                 )
        self.dconv4=nn.Sequential(ReparamConv(filters[0],filters[1],filters[0],5,3),
                                 ReparamConv(filters[0], filters[1], filters[0],5,3),
                                 )



        # self.dconv1=nn.Sequential(ReparamConv(2*filters[3],2*filters[4],2*filters[3],5,3),
        #                          ReparamConv(2*filters[3], 2*filters[4], 2*filters[3],5,3),
        #
        #                          )
        # self.dconv2=nn.Sequential(ReparamConv(2*filters[2],2*filters[3],2*filters[2],5,3),
        #                          ReparamConv(2*filters[2],2*filters[3], 2*filters[2],5,3),
        #
        #                          )
        # self.dconv3=nn.Sequential(ReparamConv(2*filters[1],2*filters[2],2*filters[1],5,3),
        #                          ReparamConv(2*filters[1],2*filters[2],2*filters[1],5,3),
        #                          )
        # self.dconv4=nn.Sequential(ReparamConv(2*filters[0],2*filters[1],2*filters[0],5,3),
        #                          ReparamConv(2*filters[0],2*filters[1], 2*filters[0],5,3),
        #                          )

        #self.transformer = BottomTransformer(3, 14, filters[4], 2, filters[4], 1, 8)#单尺度GFT

        #self.aspp = My_ASPP(filters[4], filters[4])
        self.pyramidpool=PyramidPool()
        self.transformer=BottomTransformer(3, 14, sum(filters),  2, filters[4],1, 8)#多尺度GFT

        # self.fuse1 = connectionfuse(2*filters[0],filters[0])
        # self.fuse2 = connectionfuse(2*filters[1], filters[1])
        # self.fuse3 = connectionfuse(2*filters[2], filters[2])
        # self.fuse4 = connectionfuse(2*filters[3], filters[3])

        self.up1 = nn.Sequential(nn.BatchNorm2d(filters[4]),
                                       #nn.ReLU(inplace=True),
                                nn.GELU(),
                                #nn.Upsample(scale_factor=2,mode='bilinear'),
                                # nn.Conv2d(filters[4], 4*filters[3], 3,1,1),
                                # nn.PixelShuffle(2),
                                 nn.ConvTranspose2d(filters[4],filters[3],3,2,1,1)

        )
        self.up2 =nn.Sequential(nn.BatchNorm2d(filters[3]),
                      #nn.ReLU(inplace=True),
                      nn.GELU(),
                      #nn.Upsample(scale_factor=2, mode='bilinear'),
                      # nn.Conv2d(filters[3], 4*filters[2], 3,1,1),
                      #   nn.PixelShuffle(2),
                                nn.ConvTranspose2d(filters[3], filters[2], 3, 2, 1, 1)
                      )
        self.up3 =nn.Sequential(nn.BatchNorm2d(filters[2]),
                      #nn.ReLU(inplace=True),
                      nn.GELU(),
                      #nn.Upsample(scale_factor=2, mode='bilinear'),
                      # nn.Conv2d(filters[2], 4*filters[1], 3,1,1),
                      #           nn.PixelShuffle(2),
                                nn.ConvTranspose2d(filters[2], filters[1], 3, 2, 1, 1)
                      )
        self.up4 =nn.Sequential(nn.BatchNorm2d(filters[1]),
                      #nn.ReLU(inplace=True),
                      nn.GELU(),
                      #nn.Upsample(scale_factor=2, mode='bilinear'),
                      # nn.Conv2d(filters[1], 4*filters[0], 3,1,1),
                      # nn.PixelShuffle(2),
                                nn.ConvTranspose2d(filters[1], filters[0], 3, 2, 1, 1)
                      )


        # self.up1 = nn.Sequential(nn.BatchNorm2d(sum(filters)),
        #                                nn.ReLU(inplace=True),
        #                               nn.Upsample(scale_factor=2,mode='bilinear'),
        #                                nn.Conv2d(sum(filters),filters[3] , 1),
        # )
        # self.up2 =nn.Sequential(nn.BatchNorm2d(2*filters[3]),
        #               nn.ReLU(inplace=True),
        #               nn.Upsample(scale_factor=2, mode='bilinear'),
        #               nn.Conv2d(2*filters[3], filters[2], 1),
        #               )
        # self.up3 =nn.Sequential(nn.BatchNorm2d( 2*filters[2]),
        #               nn.ReLU(inplace=True),
        #               nn.Upsample(scale_factor=2, mode='bilinear'),
        #               nn.Conv2d(2*filters[2], filters[1], 1),
        #               )
        # self.up4 =nn.Sequential(nn.BatchNorm2d(2*filters[1]),
        #               nn.ReLU(inplace=True),
        #               nn.Upsample(scale_factor=2, mode='bilinear'),
        #               nn.Conv2d(2*filters[1], filters[0], 1),
        #               )

        self.skip1 = M2Skip([filters[2],filters[3]],'bottom')
        self.skip2 = M3Skip([filters[1],filters[2],filters[3]])
        self.skip3 = M3Skip([filters[0],filters[1],filters[2]])
        self.skip4 = M2Skip([filters[0], filters[1]],'top')

        self.natt1 = NeighborhoodTransformer(3,32,filters[3],filters[3],1,[3,5],8)
        self.natt2 = NeighborhoodTransformer(3, 64, filters[2], filters[2], 1,[3,5],4)
        self.natt3 = NeighborhoodTransformer(3, 128, filters[1], filters[1], 1,[3,5],4)
        self.natt4 = NeighborhoodTransformer(3, 256, filters[0], filters[0], 1,[3,5],2)

        #self.proj=nn.Conv2d(filters[4],sum(filters),1)

        self.output_layer = nn.Conv2d(filters[0], n_classes, 1)

        #self.output_layer = nn.Conv2d(2*filters[0], n_classes, 1)

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()

    def forward(self, x):
        #x=self.stem(x)

        x1 = self.conv1(x)#16,224,224
        #x1 = self.eca1(x1)
        x_down1=self.down1(x1)#16,112,112
        x2 = self.conv2(x_down1)#32,112,112
        #x2 = self.eca2(x2)
        x_down2 = self.down2(x2)#32,56,56
        x3 = self.conv3(x_down2)#64,56,56
        #x3 = self.eca3(x3)
        x_down3 = self.down3(x3)#64,28,28
        x4 = self.conv4(x_down3)#128,28,28
        #x4 = self.eca4(x4)
        x_down4 = self.down4(x4)#128,14,14


        x5=self.transformer(self.pyramidpool(x1,x2,x3,x4,x_down4))
        #x5 = self.transformer(self.proj(x_down4))

        #x5 = self.aspp(x_down4)
        #x5=torch.cat([self.aspp(x_down4),self.transformer(self.pyramidpool(x1,x2,x3,x4,x_down4))], dim=1)#256,16,16
        #x5=self.transformer(x_down4)


        #x_up1=self.up1(x5)#128,32,32

        #总结构
        # x_skip1=self.skip1(x3,x4)
        # x_skip2 = self.skip2(x2, x3,x4)
        # x_skip3 = self.skip3(x1, x2,x3)
        # x_skip4 = self.skip4(x1, x2)
        #
        # x46 = self.natt1(x_skip1)
        # x37 = self.natt2(x_skip2)
        # x28 = self.natt3(x_skip3)
        # x19 = self.natt4(x_skip4)
        #
        # x6 = self.dconv1(self.up1(x5)+x46)
        # x7 = self.dconv2(self.up2(x6) + x37)
        # x8 = self.dconv3(self.up3(x7) + x28)
        # x9 = self.dconv4(self.up4(x8) + x19)
##################################################LFT消融
        # x46 = self.natt1(x4)
        # x37 = self.natt2(x3)
        # x28 = self.natt3(x2)
        # x19 = self.natt4(x1)
        # x_skip1=self.skip1(x3,x4)
        # x_skip2 = self.skip2(x2, x3,x4)
        # x_skip3 = self.skip3(x1, x2,x3)
        # x_skip4 = self.skip4(x1, x2)
        x6 = self.dconv1(self.up1(x5)+x4)
        x7 = self.dconv2(self.up2(x6) + x3)
        x8 = self.dconv3(self.up3(x7) + x2)
        x9 = self.dconv4(self.up4(x8) + x1)








        if self.deep_supervision:
            output1 = self.output_layer(x19)
            output2 = self.output_layer(x19)
            #output3 = self.output_layer(x13+x9)
            return [output1, output2]
        else:
            #out = self.output_layer(x19)#[16,2,256,256]
            out = self.output_layer(x9)
            return out
