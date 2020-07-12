import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
learn from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

torch.manual_seed(2019)

class BasicBlock(nn.Module):
    expansion = 1 # self.expansion
    # expansion usage : 
    # assuming that the input and output are of the same dimensions
    # if in_planes! = res_planes*expansion , the id highpass will be more complex,
    # only used when matching dimensions
    def __init__(self, in_planes, res_planes, p_stride=1):
        super(BasicBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels= in_planes, out_channels= res_planes,\
                    kernel_size=3, stride=p_stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features= res_planes),
            nn.ReLU(),
            
            nn.Conv2d(in_channels= res_planes, out_channels= res_planes,\
                    kernel_size=3, stride=1, padding=1, bias=False), # keep H & W
            nn.BatchNorm2d(num_features= res_planes)
        )
        
        self.shortcut = nn.Sequential() # identity
        if p_stride != 1 or in_planes != self.expansion*res_planes:
            # can't match the channel number to ele-wise add, need to map W
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*res_planes,\
                        kernel_size=1, stride=p_stride, bias=False),
                nn.BatchNorm2d(self.expansion*res_planes)
            )
        
        # for C:
        # o_shortX = 1*res_planes
        # o_res = 1*res_planes 
        # for H , W :
        # o_shortX = (i_shortX - 1)//p_s + 1
        # o_res = (i_shortX + 2*1 - 3 )//p_s + 1

    def forward(self, px_Tsor):
        px_Tsor = self.residual(px_Tsor) + self.shortcut(px_Tsor)
        px_Tsor = F.relu(px_Tsor)
        return px_Tsor
        
        


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, res_planes, p_stride=1):
        super(Bottleneck, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels= in_planes, out_channels= res_planes,\
                    kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features= res_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels= res_planes, out_channels= res_planes,\
                    kernel_size=3, stride=p_stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features= res_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=  res_planes, out_channels=self.expansion*res_planes, \
                    kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features= self.expansion*res_planes)
        )

        self.shortcut = nn.Sequential()
        if p_stride != 1 or in_planes != self.expansion*res_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * res_planes, kernel_size=1, stride=p_stride, bias=False),
                nn.BatchNorm2d(self.expansion * res_planes)
            )
        # for C:
        # o_shortX = 4*res_planes
        # o_res = 4*res_planes

    def forward(self, px_Tsor):
        px_Tsor = self.residual(px_Tsor) + self.shortcut(px_Tsor)
        px_Tsor = F.relu(px_Tsor)
        return px_Tsor


class miniResNet(nn.Module):
    def __init__(self, p_blockType, p_convX_numLst, num_classes=10):
        # block == BasicBlock / Bottleneck
        # p_convX_numLst : a list contain conv2_x ~ conv5_x 's block num
        # num_classes:final FC output layer neuron nums
        super(miniResNet, self).__init__()
        self.cur_in_planes = 64 # first after pre-conv always be 64
        # input (3, 32, 32)
        self.preconv = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels= self.cur_in_planes,\
                    kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.cur_in_planes),
            nn.ReLU(True)
        )
        # 32 + 2*1- 3 + 1 = 32
        # output: (64, 32, 32)
        self.conv2_x = self._make_layer(p_blockType = p_blockType, res_planes= 64, \
                                        p_convX_num= p_convX_numLst[0], p_1st_stride=1)
        # output: (64*expansion, 32, 32)
        self.conv3_x = self._make_layer(p_blockType = p_blockType, res_planes= 128, \
                                        p_convX_num= p_convX_numLst[1], p_1st_stride=2)
        # output: (128*expansion, 16, 16)
        self.conv4_x = self._make_layer(p_blockType = p_blockType, res_planes= 256, \
                                        p_convX_num= p_convX_numLst[2], p_1st_stride=2)
        # output: (256*expansion, 8, 8)
        self.conv5_x = self._make_layer(p_blockType = p_blockType, res_planes= 512, \
                                        p_convX_num= p_convX_numLst[3], p_1st_stride=2)
        # output: (512*expansion, 4, 4)
        self.avgpool = nn.AvgPool2d(kernel_size = 4)
        # output: (512*expansion, 1, 1)
        
        # need stretch and then send to linear
        self.logistic = nn.Sequential(
            nn.Linear(512 * p_blockType.expansion, num_classes),
            nn.Softmax(dim=-1)
        )
    
    def _make_layer(self, p_blockType, res_planes, p_convX_num, p_1st_stride):
        strides_Lst = [p_1st_stride] + [1]*(p_convX_num-1)
        Layers_Lst = []
        for stride_i in strides_Lst :
            Layers_Lst.append(p_blockType(self.cur_in_planes, res_planes, stride_i))
            self.cur_in_planes = res_planes * p_blockType.expansion
        
        return nn.Sequential(*Layers_Lst)
    
    def forward(self, px_Tsor):
        #print("input size :", px_Tsor.size())
        px_Tsor= self.preconv(px_Tsor)
        #print("afte pre-conv :", px_Tsor.size())
        px_Tsor= self.conv2_x(px_Tsor)
        #print("afte conv2_x :", px_Tsor.size())
        px_Tsor= self.conv3_x(px_Tsor)
        #print("afte conv3_x :", px_Tsor.size())
        px_Tsor= self.conv4_x(px_Tsor)
        #print("afte conv4_x :", px_Tsor.size())
        px_Tsor= self.conv5_x(px_Tsor)
        #print("afte conv5_x :", px_Tsor.size())
        px_Tsor= self.avgpool(px_Tsor)
        #print("afte avg_pool :", px_Tsor.size())

        px_Tsor= px_Tsor.view(px_Tsor.size(0), -1)
        px_Tsor= self.logistic(px_Tsor)

        return px_Tsor

def ResNet18():
    return miniResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return miniResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return miniResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return miniResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return miniResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet50()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

if __name__ == "__main__":
    test()



        





    

