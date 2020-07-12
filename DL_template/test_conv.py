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

from arch.Lenet5 import LeNet5_1998
import loader.basicLoader as bld
import custom_utils.config as ut_cfg 
import custom_utils.initializer as ut_init
import custom_utils.evaluate as ut_eval

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("please pip install tensorboard==2.0.2")

class test_config(ut_cfg.config):
    def __init__(self):
        super(test_config, self).__init__(pBs = 32, pWn = 2, p_force_cpu = False)
        self.path_save_mdroot = self.check_path_valid(os.path.join(ROOT, "outputs_convLenet"))

        self.path_save_mdid = "lenet_MNIST\convMNIST0326_051.pth"

        self.height_in = 28
        self.width_in = 28
        self.latent_num = 16*5*5
        self.class_num = 10

        self.class_name = [str(i) for i in range(10)]

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
    
    def view_pred(self, p_img_Tsor_batch_i, p_label_Tsor_batch_i, p_pred_Tensor_batch_i):
        p_img_Tsor = torchvision.utils.make_grid(
                        tensor = p_img_Tsor_batch_i, 
                        nrow= 4)
        p_img_Tsor = p_img_Tsor
        img_Arr = p_img_Tsor.cpu().numpy()

        label_Arr = p_label_Tsor_batch_i.cpu().numpy().astype(np.str).reshape(-1, 4)
        pred_Arr = p_pred_Tensor_batch_i.cpu().numpy().astype(np.str).reshape(-1, 4)
        print("Predict <-> Label:\n", 
            np.core.defchararray.add(
                np.core.defchararray.add(pred_Arr, "-"),
                label_Arr
            )) 
        fig = plt.figure("test figure", figsize = (8,8))
        plt.subplot(1,2,1)
        plt.imshow(np.transpose(img_Arr, (1, 2, 0)))
        plt.subplot(1,2,2)
        plt.imshow(np.transpose(img_Arr, (1, 2, 0)))
        plt.show()


if __name__ == "__main__":
    
    gm_cfg = test_config()
    gm_testloader = torch.utils.data.DataLoader(
        dataset = gm_cfg.create_dataset(istrain = False), 
        batch_size= gm_cfg.ld_batchsize,
        shuffle= True,
        num_workers= gm_cfg.ld_workers
    ) # 1875 * 32

    # prepare net
    gm_net = LeNet5_1998(
            gm_cfg.width_in, gm_cfg.height_in, 
            gm_cfg.latent_num, gm_cfg.class_num)
    

    gm_cfg.init_net(gm_net)
    
    gm_eval1 = ut_eval.confusion_eval(gm_cfg.class_name)
    with torch.no_grad():
        acc_Lst = [] # len(validloader) = 313; 
        for iter_idx, (img_Tsor_bacth_i, label_Tsor_bacth_i) in enumerate(gm_testloader):
            img_Tsor_bacth_i = img_Tsor_bacth_i.to(gm_cfg.device)
            label_Tsor_bacth_i = label_Tsor_bacth_i.to(gm_cfg.device)
            pred_Tsor_bacth_i = gm_net(img_Tsor_bacth_i)
            max_likeli_pred_bacth_i = torch.argmax(pred_Tsor_bacth_i,dim = -1)
            error_num = (max_likeli_pred_bacth_i - label_Tsor_bacth_i).nonzero().shape[0] 
            cur_acc = 1 - error_num / label_Tsor_bacth_i.shape[0]
            acc_Lst.append(cur_acc)
            gm_cfg.view_pred(img_Tsor_bacth_i, label_Tsor_bacth_i, max_likeli_pred_bacth_i)

            #gm_eval1.add_data(max_likeli_pred_bacth_i, label_Tsor_bacth_i)

        print("test Accuraccy:", sum(acc_Lst) / len(acc_Lst))
        #gm_eval1.view_mat("MNIST test")

