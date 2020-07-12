import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    reference
    code & para https://github.com/pytorch/examples/blob/master/vae/main.py
'''

class fcVAE(nn.Module):
    def __init__(self, latent_num):
        super(fcVAE, self).__init__()
        # In (bs, 1, 28, 28)
        self.enc_fc1 = nn.Linear(784, 400)
        self.enc_fc21 = nn.Linear(400, latent_num)
        self.enc_fc22 = nn.Linear(400, latent_num)
        self.dec_fc3 = nn.Linear(latent_num, 400)
        self.dec_fc4 = nn.Linear(400, 784)

    def encode(self, px_Tsor):
        # (Bs, 28*28)
        common_h = torch.relu(self.enc_fc1(px_Tsor))

        qmu_Tsor = self.enc_fc21(common_h)
        qlogvar_Tsor = self.enc_fc22(common_h)

        return qmu_Tsor, qlogvar_Tsor

    def reparameterize(self, pmu_Tsor, plogvar_Tsor):
        # add noise on Z, sample trick
        std_Tsor = torch.exp(0.5*plogvar_Tsor)
        # sample an vareplison
        esp =torch.randn_like(std_Tsor)
        qsampleZ_Tsor = pmu_Tsor + std_Tsor * esp
        return qsampleZ_Tsor
    
    def decode(self, pz_Tsor):
        reconh_Tsor = torch.relu(self.dec_fc3(pz_Tsor))
        reconx_Tsor = torch.sigmoid(self.dec_fc4(reconh_Tsor)) # torch.sigmoid
        return reconx_Tsor

    def forward(self, px_Tsor):
        qmu_Tsor, qlogvar_Tsor  = self.encode(px_Tsor.view(-1, 784))
        z_Tsor = self.reparameterize(qmu_Tsor, qlogvar_Tsor)
        reconx_Tsor = self.decode(z_Tsor).view(-1, 1, 28, 28)

        return reconx_Tsor, qmu_Tsor, qlogvar_Tsor




class convVAE(nn.Module):
    def __init__(self, latent_num):
        super(convVAE, self).__init__()
        # (1,28, 28) -> (16, 14, 14) -> (64, 7, 7)
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 16, \
                        kernel_size= 3, stride=2,  padding= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels= 16, out_channels= 64, \
                        kernel_size= 3, stride=2, padding= 1),
        )

        # (64,7, 7) -> (64, 5, 5) -> (64, 3, 3) -> (16, 3, 3)-> (16, 1, 1)
        self.enc_conv21 = nn.Sequential(
            nn.Conv2d(in_channels= 64, out_channels= 64, \
                        kernel_size= 3),
            nn.ReLU(),
            nn.Conv2d(in_channels= 64, out_channels= 64, \
                        kernel_size= 3),
            nn.ReLU(),
            nn.Conv2d(in_channels= 64, out_channels= 16, \
                        kernel_size= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels= 16, out_channels= latent_num, \
                        kernel_size= 3),
        )

        # (64,7, 7) -> (64, 5, 5) -> (64, 3, 3) -> (16, 3, 3)-> (16, 1, 1)
        self.enc_conv22 = nn.Sequential(
            nn.Conv2d(in_channels= 64, out_channels= 64, \
                        kernel_size= 3),
            nn.ReLU(),
            nn.Conv2d(in_channels= 64, out_channels= 64, \
                        kernel_size= 3),
            nn.ReLU(),
            nn.Conv2d(in_channels= 64, out_channels= 16, \
                        kernel_size= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels= 16, out_channels= latent_num, \
                        kernel_size= 3),
        )


        # (16, 1, 1) -> (16, 3, 3) -> (8, 7, 7) -> (3, 14, 14) -> (1, 28, 28) 
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels = latent_num, out_channels = 16, kernel_size = 3), 
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 16, out_channels = 8, kernel_size = 3, stride= 2), 
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 3, kernel_size = 2, stride= 2), 
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 3, out_channels = 1, kernel_size = 2, stride= 2), 
        )

    def encode(self, px_Tsor):
        # (Bs, 28*28)
        BS = px_Tsor.shape[0]
        common_h = torch.relu(self.enc_conv1(px_Tsor))

        qmu_Tsor = self.enc_conv21(common_h).view(BS, -1)
        qlogvar_Tsor = self.enc_conv22(common_h).view(BS, -1)

        return qmu_Tsor, qlogvar_Tsor
    
    def reparameterize(self, pmu_Tsor, plogvar_Tsor):
        # add noise on Z, sample trick
        std_Tsor = torch.exp(0.5*plogvar_Tsor)
        # sample an vareplison
        esp =torch.randn_like(std_Tsor)
        qsampleZ_Tsor = pmu_Tsor + std_Tsor * esp
        return qsampleZ_Tsor
    
    def decode(self, pz_Tsor):
        BS = pz_Tsor.shape[0]
        pz_Tsor = pz_Tsor.view(BS, -1 , 1, 1)
        qreon_Tsor = torch.sigmoid(self.dec_conv(pz_Tsor))
        return qreon_Tsor


    def forward(self, px_Tsor):
        qmu_Tsor, qlogvar_Tsor  = self.encode(px_Tsor)
        z_Tsor = self.reparameterize(qmu_Tsor, qlogvar_Tsor)
        reconx_Tsor = self.decode(z_Tsor)

        return reconx_Tsor, qmu_Tsor, qlogvar_Tsor


def check_arch_para(pNet):
        para_amount = 0
        for para_i in pNet.parameters():
            # para_i is a torch.nn.parameter.Parameter, grad 默认 True
            # print(para_i.shape, para_i.requires_grad)
            para_amount+= para_i.numel()
        print("[net info] para_amount=%d"%para_amount)


if __name__ == "__main__":
    
    x_Tsor = torch.rand(5,1,28,28)

    print("Input shape:\n",x_Tsor.shape)
    # gm_net = convVAE(16) # 167464
    gm_net = fcVAE(16)     # 648016
    check_arch_para(gm_net)

    recon_Tsor, mu, logvar = gm_net(x_Tsor)

    print("checking Output shape:\n", recon_Tsor.shape, "\t", mu.shape, "\t",logvar.shape)

