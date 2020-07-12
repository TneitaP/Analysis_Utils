import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    expansion = 1 # self.expansion
    def __init__(self, in_planes, res_planes, p_stride=1):
        super(ResBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels= in_planes, out_channels= res_planes,\
                    kernel_size=1, stride=p_stride, bias=False),
            nn.BatchNorm1d(num_features= res_planes),
            #nn.ReLU(),
            nn.PReLU(),
            nn.Conv1d(in_channels= res_planes, out_channels= res_planes,\
                    kernel_size=1, bias=False), # keep H & W
            nn.BatchNorm1d(num_features= res_planes)
        )
        self.shortcut = nn.Sequential() # identity
        if p_stride != 1 or in_planes != self.expansion*res_planes:
            # can't match the channel number to ele-wise add, need to map W
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*res_planes,\
                        kernel_size=1, stride=p_stride, bias=False),
                nn.BatchNorm1d(self.expansion*res_planes)
            )
    
    def forward(self, px_Tsor):
        qx_Tsor = self.residual(px_Tsor) + self.shortcut(px_Tsor)
        qx_Tsor = F.relu(qx_Tsor)
        return qx_Tsor

class GetMax_Flatten(nn.Module):
    def forward(self, x):
        #input (BS, C, [2]H= 1024/4096)
        qx = torch.max(x, dim=2, keepdim=True)[0]
        #qx (BS, C, [2]H= 1)
        qx = qx.view(qx.size(0), -1)
        #qx (BS, C*H)
        return qx

class UnFlatten(nn.Module):
    def __init__(self, channel):
        super(UnFlatten, self).__init__()
        self.channel = channel
    def forward(self, x):
        return x.view(x.size(0), self.channel, -1)

class PointNetVAE_max(nn.Module):
    def __init__(self, hidden_dim=512, z_dim=32):
        super(PointNetVAE_max, self).__init__()
        #input size (N, C = 3, H = 1024 or 4096)
        '''
            common Encoder to extract feature
        '''
        self.conv_com = nn.Sequential(
            nn.Conv1d(in_channels= 3, out_channels= 64, kernel_size= 1),
            nn.BatchNorm1d(64),
            #nn.ReLU(),
            nn.PReLU(),
            # (64,1024)
            ResBlock(in_planes= 64, res_planes= 64),
            ResBlock(64,64),
            ResBlock(in_planes= 64, res_planes= 128),
            # (128,1024)
            ResBlock(128,128),

            ResBlock(in_planes= 128, res_planes= 256),
            # (,256,1024)
            ResBlock(256,256),

            ResBlock(in_planes= 256, res_planes= 512),
            # (,512,1024)
            ResBlock(512,hidden_dim),
            # (,512,1024)
            GetMax_Flatten()
            # (,512*1)
        )
        self.mu_encoder_max = nn.Linear(in_features=hidden_dim*1, out_features= z_dim)
        self.logvar_encoder_max = nn.Linear(in_features=hidden_dim*1, out_features= z_dim)
        
        self.com_decoder = nn.Linear(in_features=z_dim, out_features= hidden_dim)
        
        self.convT_decoder = nn.Sequential(
            UnFlatten(channel= hidden_dim),
            #(BS, hidden_dim = 512, 1)
            nn.ConvTranspose1d(in_channels= 512, out_channels= 256,\
                            kernel_size= 3, stride= 3, ),
            nn.PReLU(),
            #(BS, hidden_dim = 256, 3)
            nn.ConvTranspose1d(in_channels= 256, out_channels= 128,\
                            kernel_size= 4, stride= 4, ),
            #(BS, hidden_dim = 128, 12)
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels= 128, out_channels= 64,\
                            kernel_size= 4, stride= 4, ),
            #(BS, hidden_dim = 64, 48)
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels= 32,\
                            kernel_size= 4, stride= 4, ),
            #(BS, hidden_dim = 32, 192)
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels= 3,\
                            kernel_size= 4, stride= 4, ),
            #(BS, hidden_dim = 3, 768)
        )
        self.linear_decoder = nn.Sequential(
            #(BS, hidden_dim = 512*1*1)
            nn.Linear(in_features=512*1, out_features= 1024),
            nn.Dropout(),
            nn.PReLU(),
            nn.Linear(in_features=1024, out_features= 768),
            nn.Dropout(),
            nn.PReLU(),
            nn.Linear(in_features=768, out_features= 256*3),#768
            #(BS, 768)
            UnFlatten(channel= 3),
            #(BS, 3, 256)
        )



    def encode(self, px_Tsor):
        hidden_Tsor = self.conv_com(px_Tsor)
        # after conv_com, hidden_Tsor.Size([BS, hidden_dim*1]), for max only
        qmu_Tsor = self.mu_encoder_max(hidden_Tsor)
        qlogvar_Tsor = self.logvar_encoder_max(hidden_Tsor)
        # expect [BS, 32]
        return qmu_Tsor, qlogvar_Tsor
    
    def reparametrize(self, pmu_Tsor, plogvar_Tsor):
        # add noise on Z, sample trick
        std_Tsor = plogvar_Tsor.mul(0.5).exp_() # biaozhuncha
        # sample an vareplison
        esp = torch.randn(*pmu_Tsor.size())
        qsampleZ_Tsor = pmu_Tsor + std_Tsor * esp
        return qsampleZ_Tsor

    def decode(self, pz_Tsor):
        reconh_Tsor = self.com_decoder(pz_Tsor)
        X_convT_Tsor = self.convT_decoder(reconh_Tsor)
        X_LinAdd_Tsor = self.linear_decoder(reconh_Tsor)
        # X_convT_Tsor.Size([BS, 3, 768]); X_LinAdd_Tsor.Size([BS, 3, 256])
        q_reconX_Tsor = torch.cat([X_LinAdd_Tsor, X_convT_Tsor], dim = 2)
        return q_reconX_Tsor

    def forward(self, px_Tsor):
        qmu_Tsor, qlogvar_Tsor = self.encode(px_Tsor)
        qsampleZ_Tsor = self.reparametrize(qmu_Tsor, qlogvar_Tsor)
        q_reconX_Tsor= self.decode(qsampleZ_Tsor)
        return q_reconX_Tsor, qmu_Tsor, qlogvar_Tsor



def test_output():
    t_net = PointNetVAE_max()
    t_input_Tsor = torch.randn(10, 3, 1024)#.cuda()
    reconX_Tsor, mu_Tsor, logvar_Tsor = t_net(t_input_Tsor)
    print(reconX_Tsor.size())
    print(mu_Tsor.size())
    print(logvar_Tsor.size())
    print(t_net.state_dict().keys())
    
    param_Tsor = list(t_net.parameters())
    print(param_Tsor[1])

def test_saveload():
    save_dir = "C:\\Users\\TneitaP\\Desktop\\pcd_generate"
    model_dir = os.path.join(save_dir, 'params.pth')
    # 1. 训练结束， 保存当前 model_1 的权重字典
    torch.save(t_net.state_dict(), model_dir)
    
    # 2. 重新初始化一个 model_2: 检验的确权重初值与之前model_1 不同
    t_net2 = PointNetVAE_max()
    param_Tsor2_before = list(t_net2.parameters())
    print(param_Tsor2_before[1]==param_Tsor[1])
    # 3. 将保存的 model_1 权重字典 加载到 model_2 上： 检验的确权重初值与之前model_1 同
    t_net2.load_state_dict(torch.load(model_dir))
    param_Tsor2_after = list(t_net2.parameters())
    print(param_Tsor2_after[1]==param_Tsor[1])

if __name__ == "__main__":
    save_dir = "C:\\Users\\TneitaP\\Desktop\\pcd_generate"
    model_dir = os.path.join(save_dir, 'params.pth')
    

    
    
    