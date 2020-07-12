import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import time 
import os 
import sys
ROOT = os.getcwd()
sys.path.append(ROOT)

# from arch.vae import fcVAE
from arch.vae import convVAE

import loader.basicLoader as bld
import custom_utils.config as ut_cfg 
import custom_utils.initializer as ut_init
import custom_utils.evaluate as ut_eval

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("please pip install tensorboard==2.0.2")


def view_tensor(p_img_Tsor):
    # p_img_Tsor = p_img_Tsor / 2 + 0.5     # unnormalize
    if p_img_Tsor.is_cuda:
        p_img_Tsor = p_img_Tsor.cpu()
    if p_img_Tsor.requires_grad:
        p_img_Tsor = p_img_Tsor.detach()
    img_Arr = p_img_Tsor.numpy()
    plt.imshow(np.transpose(img_Arr, (1, 2, 0)))
    plt.show()

class test_config(ut_cfg.config):
    def __init__(self):
        super(test_config, self).__init__(pBs = 32, pWn = 2, p_force_cpu = False)
        self.path_save_mdroot = self.check_path_valid(os.path.join(ROOT, "outputs"))

        self.path_save_mdid = "convVaeMNIST0327_500.pth"

        self.height_in = 28
        self.width_in = 28
        self.latent_num = 16

        self.method_init = "preTrain" #"preTrain" #"kaming"#"xavier"

        self.dtroot = os.path.join(ROOT, "datasets")
    
    def create_dataset(self, istrain):
        q_dataset = bld.toy_dataset(bld.ToyDatasetName.MNIST, 
            pRoot = self.dtroot, pIstrain = istrain, 
            pTransform= None, pDownload= True
        )
        return q_dataset

    def init_net(self, pNet):
        assert self.method_init == "preTrain"
        pretrained_weight_path = os.path.join(self.path_save_mdroot, self.path_save_mdid)
        pNet.load_state_dict(torch.load(pretrained_weight_path))

        pNet.to(self.device).eval()


if __name__ == "__main__":
    
    gm_cfg = test_config()
    gm_testloader = torch.utils.data.DataLoader(
        dataset = gm_cfg.create_dataset(istrain = False), 
        batch_size= gm_cfg.ld_batchsize,
        shuffle= False,
        num_workers= gm_cfg.ld_workers
    ) # 1875 * 32

    # prepare net
    gm_net = convVAE(gm_cfg.latent_num)

    gm_cfg.init_net(gm_net)

    with torch.no_grad():
        for iter_idx, (img_Tsor_bacth_i, _) in enumerate(gm_testloader):
            img_Tsor_bacth_i = img_Tsor_bacth_i.to(gm_cfg.device)
            recon_Tsor_bacth_i, _, _ = gm_net(img_Tsor_bacth_i)
            view_tensor(torchvision.utils.make_grid(
                        tensor = recon_Tsor_bacth_i, 
                        nrow= 4)
            )