import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Reference:
[1] Going Deeper withConvolutions, 6.67% test error，2014.9
[2] Batch Normalization:Accelerating Deep Network Training \
    by Reducing Internal Covariate Shift， 4.8% test error, 2015

[3] Rethinking theInception Architecture for Computer Vision, \
3.5%test error，2015.12
'''

torch.manual_seed(2019)


class Inception_V1(nn.Module):
    def __init__(self, in_planes, \
                n1x1, n3x3reduce, n3x3, n5x5reduce, n5x5, pool_planes):
        super(Inception_V1, self).__init__()

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels= in_planes, out_channels= n1x1, \
                kernel_size=1),
            nn.ReLU()
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels= in_planes, out_channels= n3x3reduce, \
                    kernel_size= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels= n3x3reduce, out_channels= n3x3,\
                    kernel_size = 3,padding=1),
            nn.ReLU()
        )

        
        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels= in_planes, out_channels= n5x5reduce,\
                    kernel_size = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels= n5x5reduce, out_channels= n5x5,\
                    kernel_size = 5 ,padding=2),
            nn.ReLU()
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride=1, padding=1),
            nn.Conv2d(in_channels= in_planes, out_channels=pool_planes, \
                    kernel_size=1),
            nn.ReLU()
        )

    def forward(self, px_Tsor):
        print("inpt size:", px_Tsor.size())
        y1_Tsor = self.b1(px_Tsor)
        print("branch1_out size:", y1_Tsor.size())
        y2_Tsor = self.b2(px_Tsor)
        print("branch2_out size:", y2_Tsor.size())
        y3_Tsor = self.b3(px_Tsor)
        print("branch3_out size:", y3_Tsor.size())
        y4_Tsor = self.b4(px_Tsor)
        print("branch4_out size:", y4_Tsor.size())
        
        qx_Tsor = torch.cat([y1_Tsor,y2_Tsor,y3_Tsor,y4_Tsor], 1)
        print("total_out size:", qx_Tsor.size())
        return qx_Tsor

class Inception_V2(nn.Module):
    def __init__(self, in_planes, \
                n1x1, n3x3reduce, n3x3, n5x5reduce, n5x5, pool_planes):
        super(Inception_V2, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=n1x1, \
                    kernel_size=1),
            nn.BatchNorm2d(num_features= n1x1),
            nn.ReLU()
        )
        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels= in_planes, out_channels= n3x3reduce, \
                    kernel_size= 1),
            nn.BatchNorm2d(num_features= n3x3reduce),
            nn.ReLU(),
            nn.Conv2d(in_channels= n3x3reduce, out_channels= n3x3,\
                    kernel_size = 3,padding=1),
            nn.BatchNorm2d(num_features= n3x3),
            nn.ReLU()
        )
        
        # V1:1x1 conv -> 5x5 conv branch
        # V2: 1x1 conv -> 3x3 conv branch -> 3x3 conv branch
        # less parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels= in_planes, out_channels= n5x5reduce,\
                    kernel_size = 1),
            nn.BatchNorm2d(num_features= n5x5reduce),
            nn.ReLU(),
            nn.Conv2d(in_channels= n5x5reduce, out_channels= n5x5,\
                    kernel_size = 3 ,padding=1),
            nn.BatchNorm2d(num_features= n5x5),
            nn.ReLU(),
            nn.Conv2d(in_channels= n5x5, out_channels= n5x5,\
                    kernel_size = 3 ,padding=1),
            nn.BatchNorm2d(num_features= n5x5),
            nn.ReLU()
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride=1, padding=1),
            nn.Conv2d(in_channels= in_planes, out_channels=pool_planes, \
                    kernel_size=1),
            nn.BatchNorm2d(num_features= pool_planes),
            nn.ReLU()
        )
    
    def forward(self, px_Tsor):
        #print("inpt size:", px_Tsor.size())
        y1_Tsor = self.b1(px_Tsor)
        #print("branch1_out size:", y1_Tsor.size())
        y2_Tsor = self.b2(px_Tsor)
        #print("branch2_out size:", y2_Tsor.size())
        y3_Tsor = self.b3(px_Tsor)
        #print("branch3_out size:", y3_Tsor.size())
        y4_Tsor = self.b4(px_Tsor)
        #print("branch4_out size:", y4_Tsor.size())
        
        qx_Tsor = torch.cat([y1_Tsor,y2_Tsor,y3_Tsor,y4_Tsor], 1)
        #print("total_out size:", qx_Tsor.size())
        return qx_Tsor


'''
V3
self.conv1_1 = nn.Conv2d(in_channels= 1, out_channels= 1,\
                        kernel_size= (3,1), stride= 1, padding=0)
self.conv1_2 = nn.Conv2d(in_channels= 1, out_channels= 1,\
                        kernel_size= (1,3), stride= 1, padding=0)
'''

class miniGoogLeNet(nn.Module):
    # using Inception_V2
    # input : 3@ 32 x 32
    def __init__(self):
        super(miniGoogLeNet, self).__init__()
        self.preconv = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels= 192,\
                    kernel_size= 3,padding=1),
            # 32 + 2*1 -3  + 1= 32
            # output: (192, 32, 32)
            nn.BatchNorm2d(192),
            nn.ReLU()
            )
        self.a3 = Inception_V2(in_planes= 192,  n1x1= 64,  \
                    n3x3reduce= 96, n3x3= 128, \
                    n5x5reduce= 16, n5x5= 32, pool_planes= 32)
        # input_channel = in_planes(192)
        # out_channel = n1x1 + n3x3 + n5x5 + pool_planes  = 64+128+32+32=256
        # (256, 32, 32)
        self.b3 = Inception_V2(256, 128, \
                            128, 192, \
                            32, 96, 64)
        # 128 + 192 + 96 + 64 = 480
        # (480, 32, 32)
        self.maxpool = nn.MaxPool2d(kernel_size= 3, stride=2, padding=1)
        # (32 +2*1 - 3) //2 +1 = 16
        # (480, 16, 16)

        self.a4 = Inception_V2(480, 192,  \
                            96, 208, \
                            16,  48,  64)
        # 192 + 208 + 48 + 64 = 512
        # (512, 16, 16)
        self.b4 = Inception_V2(512, 160, \
                            112, 224, \
                            24,  64,  64)
        # (512, 16,16)
        self.c4 = Inception_V2(512, 128, \
                            128, 256, \
                            24,  64,  64)
        # (512, 16, 16)
        self.d4 = Inception_V2(512, 112, \
                            144, 288, \
                            32,  64,  64)
        # (528, 16, 16)
        self.e4 = Inception_V2(528, 256, \
                            160, 320, \
                            32, 128, 128)
        # (832, 16, 16)
        '''
        #same as before
        self.maxpool = nn.MaxPool2d(kernel_size= 3, stride=2, padding=1)
        '''
        # (16 +2 *1 - 3)//2 + 1 = 8
        # (832, 8 , 8)

        self.a5 = Inception_V2(832, 256, \
                            160, 320, \
                            32, 128, 128)
        # (832, 8, 8)
        self.b5 = Inception_V2(832, 384, \
                            192, 384, \
                            48, 128, 128)
        # (1024, 8, 8)
        self.avgpool = nn.AvgPool2d(kernel_size= 8, stride=1)
        # 8 - 8 + 1 = 1
        self.logistic = nn.Sequential(
            nn.Linear(1024, 10),
            nn.Softmax(dim = -1)
        )
    
    def forward(self, px_Tsor):
        print("inpt size:", px_Tsor.size())
        px_Tsor = self.preconv(px_Tsor)
        print("after pre-conv size: ", px_Tsor.size())
        px_Tsor = self.a3(px_Tsor)
        px_Tsor = self.b3(px_Tsor)
        print("after 3rd incept size: ", px_Tsor.size())
        px_Tsor = self.maxpool(px_Tsor)
        print("after 3rd pool size: ", px_Tsor.size())
        px_Tsor = self.a4(px_Tsor)
        px_Tsor = self.b4(px_Tsor)
        px_Tsor = self.c4(px_Tsor)
        px_Tsor = self.d4(px_Tsor)
        px_Tsor = self.e4(px_Tsor)
        print("after 4th incept size: ", px_Tsor.size())
        px_Tsor = self.maxpool(px_Tsor)
        print("after 4th pool size: ", px_Tsor.size())

        px_Tsor = self.a5(px_Tsor)
        px_Tsor = self.b5(px_Tsor)
        print("after 5th incept size: ", px_Tsor.size())
        px_Tsor = self.avgpool(px_Tsor)
        print("after 5th pool size: ", px_Tsor.size())

        px_Tsor = px_Tsor.view(px_Tsor.size(0), -1)
        print("after strech size: ", px_Tsor.size())
        px_Tsor = self.logistic(px_Tsor)
        return px_Tsor


if __name__ == "__main__":
    t_module = Inception_V2(in_planes= 192,  n1x1= 64,  \
                    n3x3reduce= 96, n3x3= 128, \
                    n5x5reduce= 16, n5x5= 32, pool_planes= 32)
    print(t_module)

    t_input_Tsor = torch.randn(2, 192, 227, 227)
    t_output_Tsor = t_module(t_input_Tsor)
    print("test_module_output:\n", t_output_Tsor.shape)


    t_net = miniGoogLeNet()
    print(t_net)

    t_input_Tsor = torch.randn(2, 3, 32, 32)
    t_output_Tsor = t_net(t_input_Tsor)
    print("test_goolgle_output:\n", t_output_Tsor.shape)




    