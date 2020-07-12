import torch
import torch.nn as nn
import torch.nn.functional as F 


class qpos2force(nn.Module):
    def __init__(self):
        super(qpos2force, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(24, 128), 
            nn.ReLU(), 
            nn.Linear(128, 256), 
            nn.ReLU(), 
            nn.Linear(256, 1024), 
            nn.ReLU(), 
            nn.Linear(1024, 256), 
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 24),
            nn.Tanh()
        )

    def forward(self, qpos_Tsor):
        '''
        input: qpos_Tsor shape (Bs, 24);
        output: force_Tsor shape (Bs, 24);
        '''
        force_Tsor = self.mlp(qpos_Tsor)
        return force_Tsor


if __name__ == "__main__":
    gm_net = qpos2force()

    in_Tsor = torch.rand(5, 24)

    out_Tsor = gm_net(in_Tsor)

    print("output shape: ", out_Tsor.shape)