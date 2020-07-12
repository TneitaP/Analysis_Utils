import torch
import torch.nn as nn
import torch.nn.functional as F

#torch.manual_seed(2019)

class LeNet5_1998(nn.Module):
    # inherit from the "nn.Module"
    def __init__(self, width_in, height_in, latent_num, class_num):
        '''
        width_in = height_in = 32
        latent_num = 16*5*5
        '''
        super(LeNet5_1998, self).__init__()#super init
        # input size 32 x 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 6, \
                        kernel_size= 3),
            # o = (i + pad*2 - k)//stride + 1
            # 32 - 5+ 1 = 28
            nn.ReLU(),
            # output (6,28,28)
            nn.MaxPool2d(kernel_size=2)
            # o = (i + pad*2 - k)//stride + 1
            # **pool-default: stride = k , pad = 0
            #output (6, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels= 6, out_channels= 16, \
                        kernel_size= 3),
            #output (16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            #output (16, 5, 5)
        )

        self.fc = nn.Sequential(
            # 1st FC from stretched conv, dim_in = C@H*W
            nn.Linear(latent_num ,120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, class_num),
            nn.Softmax(dim=-1) # normalize to 1, log to accelerate
            # LogSoftmax get faster
        )
    
    def forward(self, px_Tsor):
        #print("inpt size:", px_Tsor.size())
        px_Tsor = self.conv1(px_Tsor)
        px_Tsor = self.conv2(px_Tsor)
        #print("after conv2:", px_Tsor.size())
        px_Tsor = px_Tsor.view(px_Tsor.size(0), -1) # -1 means auto, =num_flat_features
        #print("after stretching:", px_Tsor.size())
        px_Tsor = self.fc(px_Tsor)
        return px_Tsor




if __name__ == "__main__":
    t_net = LeNet5_1998()
    print("Lenet-5 Architexture:\n ", t_net)

    t_input_Tsor = torch.randn(2, 1, 32, 32)
    t_output_Tsor = t_net(t_input_Tsor)
    print("test_output:\n", t_output_Tsor)
