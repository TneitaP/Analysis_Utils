import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2019)

class AlexNet_2012(nn.Module):

    def __init__(self):
        super(AlexNet_2012, self).__init__()#super init
        # input size 3 @ 227 x 227
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels= 96,\
                    kernel_size=11, stride=4),
            # (227 - 11)//4 + 1 = 55
            # output (24,55,55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5),# same shape as input
            nn.MaxPool2d(kernel_size=3, stride=2)
            # (55 - 3)//2 + 1 = 27
            # output (96,27 ,27)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels= 96, out_channels= 256,\
                    kernel_size = 5, padding=2),
            # 27 + 2*2 - 5 + 1 = 27
            # output: (256, 27, 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5),# same shape as input
            nn.MaxPool2d(kernel_size=3, stride=2)
            # (27 - 3)//2 + 1 = 13
            # output: (256, 13, 13)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels= 256, out_channels= 384, \
                    kernel_size = 3 , padding=1),
            # 13 + 2*1 - 3 + 1 = 13
            #output: (384, 13,13)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5)# same shape as input
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels= 384, out_channels= 384, \
                    kernel_size = 3 , padding=1),
            # 13 + 2*1 - 3 + 1 = 13
            #output: (384, 13,13)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5)# same shape as input
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels= 384, out_channels= 256, \
                    kernel_size = 3 , padding=1),
            # 13 + 2*1 - 3 + 1 = 13
            #output: (256, 13,13)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5),# same shape as input
            nn.MaxPool2d(kernel_size = 3 , stride=2)
            # (13  - 3)//2 + 1 = 6
            #output: (256, 6,6)
        )

        self.drop_fc = nn.Sequential(
            # input strech 9216
            nn.Linear(256*6*6, 4096),
            nn.Dropout(p= 0.5),
            nn.Linear(4096,4096),
            nn.Dropout(p= 0.5),
            nn.Linear(4096,1000),
            nn.Softmax(dim=-1)
        )

    def forward(self, px_Tsor):
        print("inpt size:", px_Tsor.size())
        px_Tsor = self.conv1(px_Tsor)
        print("after conv1:", px_Tsor.size())
        px_Tsor = self.conv2(px_Tsor)
        print("after conv2:", px_Tsor.size())
        px_Tsor = self.conv3(px_Tsor)
        print("after conv3:", px_Tsor.size())
        px_Tsor = self.conv4(px_Tsor)
        print("after conv4:", px_Tsor.size())
        px_Tsor = self.conv5(px_Tsor)
        print("after conv5:", px_Tsor.size())
        px_Tsor = px_Tsor.view(px_Tsor.size(0), -1)
        px_Tsor = self.drop_fc(px_Tsor)
        return px_Tsor


if __name__ == "__main__":
    t_net = AlexNet_2012()
    print("Lenet-5 Architexture:\n ", t_net)

    t_input_Tsor = torch.randn(2, 3, 227, 227)
    t_output_Tsor = t_net(t_input_Tsor)
    print("test_output:\n", t_output_Tsor.shape)



