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

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("please pip install tensorboard==2.0.2")


############################ Loss for VAE #########################################
def BCE_loss(precon_Tsor, px_Tsor):
    '''
    Binary Cross Entropy
    '''
    BS= px_Tsor.shape[0]
    qbce = F.binary_cross_entropy(precon_Tsor.view(BS, -1), px_Tsor.view(BS, -1), reduction='sum')
    return qbce

def KLD_loss(pmu_Tsor, plogvar_Tsor):
    '''
    KL divergence loss
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    '''
    qkld = -0.5 * torch.sum(1 + plogvar_Tsor - pmu_Tsor.pow(2) - plogvar_Tsor.exp())
    return qkld
###################################################################################

class train_config(ut_cfg.config):
    def __init__(self):
        super(train_config, self).__init__(pBs = 64, pWn = 2, p_force_cpu = False)
        self.path_save_mdroot = self.check_path_valid(os.path.join(ROOT, "outputs"))
        localtime = time.localtime(time.time())
        self.path_save_mdid = "convVaeMNIST" + "%02d%02d"%(localtime.tm_mon, localtime.tm_mday)

        self.save_epoch_begin = 40
        self.save_epoch_interval = 50

        self.log_epoch_txt = open(os.path.join(self.path_save_mdroot, "convVae_epoch_loss_log.txt"), 'a+')
        self.writer = SummaryWriter(log_dir=os.path.join(self.path_save_mdroot, "board"))

        self.height_in = 28
        self.width_in = 28
        self.latent_num = 16

        self.method_init = "xavier" #"preTrain" #"kaming"#"xavier"
        self.training_epoch_amount = 500

        self.dtroot = os.path.join(ROOT, "datasets")

        self.opt_baseLr = 1e-3
        self.opt_bata1 = 0.9
        self.opt_weightdecay = 3e-6

    def init_net(self, pNet):
        if self.method_init == "xavier":
            ut_init.init_xavier(pNet)
        elif self.method_init == "kaiming":
            ut_init.init_kaiming(pNet)
        
        elif self.method_init == "preTrain":
            assert self.preTrain_model_path is not None, "weight path ungiven"
            pNet.load_state_dict(torch.load(self.preTrain_model_path))

        pNet.to(self.device).train()
    
    def create_dataset(self, istrain):
        q_dataset = bld.toy_dataset(bld.ToyDatasetName.MNIST, 
            pRoot = self.dtroot, pIstrain = istrain, 
            pTransform= None, pDownload= True
        )
        return q_dataset
    
    def name_save_model(self, save_mode, epochX = None):
        model_filename = self.path_save_mdid
        if save_mode == "processing":
            assert epochX is not None, "miss the epoch info" 
            model_filename += "_%03d"%(epochX) + ".pth"
        elif save_mode == "ending":
            model_filename += "_%03d"%(self.training_epoch_amount) + ".pth"
        elif save_mode == "interrupt":
            model_filename += "_interrupt"+ ".pth"
        assert os.path.splitext(model_filename)[-1] == ".pth"
        q_abs_path = os.path.join(self.path_save_mdroot, model_filename)
        return q_abs_path
    
    def log_in_file(self, *print_paras):
        for para_i in print_paras:
            print(para_i, end= "")
            print(para_i, end= "", file = self.log_epoch_txt)
        print("")
        print("", file = self.log_epoch_txt)
    
    def log_in_board(self, chartname,data_Dic, epoch):
        # for key_i, val_i in data_Dic:
        self.writer.add_scalars(chartname, 
            data_Dic, epoch)
    
    def validate(self, pNet, p_epoch):
        # use the BCE loss & KLD loss to validate the AE performance
        valid_dataset = self.create_dataset(istrain=False)
        validloader = torch.utils.data.DataLoader(
            dataset = valid_dataset, 
            batch_size= self.ld_batchsize,
            shuffle= False,
            num_workers= self.ld_workers
        )

        
        bce_Lst = []
        kld_Lst = []
        for iter_idx, (img_Tsor_bacth_i, label_Tsor_bacth_i) in enumerate(validloader):
            BS = len(img_Tsor_bacth_i)
            img_Tsor_bacth_i = img_Tsor_bacth_i.to(self.device)
            recon_Tsor_bacth_i, mu_Tsor_bacth_i, logvar_Tsor_bacth_i = gm_net(img_Tsor_bacth_i)
            cur_bce = BCE_loss(recon_Tsor_bacth_i, img_Tsor_bacth_i)
            cur_kld = KLD_loss(mu_Tsor_bacth_i, logvar_Tsor_bacth_i)
            bce_Lst.append(cur_bce)
            kld_Lst.append(cur_kld)
            if iter_idx == 1:
                w_layout = int(np.sqrt(BS))
            
                view_x_Tsor = torchvision.utils.make_grid(tensor = img_Tsor_bacth_i, nrow= w_layout)
                view_recon_Tsor = torchvision.utils.make_grid(tensor = recon_Tsor_bacth_i, nrow= w_layout)
                
                self.writer.add_image("CONVinput image", view_x_Tsor, p_epoch)
                self.writer.add_image("CONVdecode image", view_recon_Tsor, p_epoch)
                # plt.clf() # clear figure



        # more care about the reconstrustion quality
        return sum(bce_Lst) / len(bce_Lst)







if __name__ == "__main__":
    
    gm_cfg = train_config()
    
    # prepare data
    gm_trainloader = torch.utils.data.DataLoader(
        dataset = gm_cfg.create_dataset(istrain = True), 
        batch_size= gm_cfg.ld_batchsize,
        shuffle= True,
        num_workers= gm_cfg.ld_workers
    ) # 1875 * 32

    # prepare net
    gm_net = convVAE(gm_cfg.latent_num)
    
    gm_cfg.init_net(gm_net)

    # optimizer & scheduler
    gm_optimizer = optim.Adam(
        params = gm_net.parameters(),
        lr = gm_cfg.opt_baseLr,
        betas= (gm_cfg.opt_bata1, 0.99),
        weight_decay = gm_cfg.opt_weightdecay
    )

    gm_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = gm_optimizer,
        mode='min',
        factor=0.8, patience=5, verbose=True, 
        threshold=0.0001, threshold_mode='rel', 
        cooldown=0, min_lr=0, eps=1e-08
    )


    loss_an_epoch_Lst = []
    bce_an_epoch_Lst = []
    kld_an_epoch_Lst = []
    try:
        print("Train_Begin".center(40, "*"))
        gm_cfg.check_arch_para(gm_net)
        gm_cfg.log_in_file("net_id = ", gm_cfg.path_save_mdid, ", batchsize = ", gm_cfg.ld_batchsize, ", workers = ", gm_cfg.ld_workers)
        gm_cfg.log_in_file("criterion_use: BCE & KLD ", ", init: ", gm_cfg.method_init)
        for epoch_i in range(gm_cfg.training_epoch_amount):
            start=time.time()
            # single epoch
            for iter_idx, (img_Tsor_bacth_i, label_Tsor_bacth_i) in enumerate(gm_trainloader):
                # valid the data-in
                # view_tensor(torchvision.utils.make_grid(
                #                 tensor = img_Tsor_bacth_i, 
                #                 nrow= 4)
                #     )

                # train process:
                img_Tsor_bacth_i = img_Tsor_bacth_i.to(gm_cfg.device)
                
                recon_Tsor_bacth_i, mu_Tsor_bacth_i, logvar_Tsor_bacth_i = gm_net(img_Tsor_bacth_i)

                # create graph in tensorboard
                # if iter_idx == 0:
                #     gm_cfg.writer.add_graph(gm_net, img_Tsor_bacth_i)

                gm_optimizer.zero_grad() # clear old grad
                cur_bce = BCE_loss(recon_Tsor_bacth_i, img_Tsor_bacth_i)
                cur_kld = KLD_loss(mu_Tsor_bacth_i, logvar_Tsor_bacth_i)
                cur_loss = cur_bce + cur_kld

                cur_loss.backward()  ## caculate new grad; retain_graph=True
                loss_an_epoch_Lst.append(cur_loss.item())
                bce_an_epoch_Lst.append(cur_bce.item())
                kld_an_epoch_Lst.append(cur_kld.item())

                gm_optimizer.step()   ### upgrade the para() using new grad
            # end an epoch
            delta_t = (time.time()- start)/60
            avg_loss = sum(loss_an_epoch_Lst)/len(loss_an_epoch_Lst)
            avg_bce = sum(bce_an_epoch_Lst)/len(bce_an_epoch_Lst)
            avg_kld = sum(kld_an_epoch_Lst)/len(kld_an_epoch_Lst)
            gm_scheduler.step(avg_loss)

            
            gm_cfg.log_in_board( "traing loss", 
                {"convVAEavg_loss": avg_loss, 
                "convVAEavg_bce": avg_bce, 
                "convVAEavg_kld": avg_kld,
                },  epoch_i
            )

            gm_cfg.log_in_file("epoch = %03d, time_cost(min)= %2.2f, loss = %2.5f, bce = %2.5f, kld = %2.5f"
                %(epoch_i, delta_t, avg_loss, avg_bce, avg_kld)
            )
            # validate the accuracy
            vali_bce = gm_cfg.validate(gm_net, epoch_i)
            gm_cfg.log_in_board( "validate loss", 
                {"convVAEvali_bce": vali_bce, 
                },  epoch_i
            )

            loss_an_epoch_Lst.clear()

            if (epoch_i >gm_cfg.save_epoch_begin and epoch_i %gm_cfg.save_epoch_interval == 1):
                # save weight at regular interval
                torch.save(obj = gm_net.state_dict(), 
                    f = gm_cfg.name_save_model("processing", epoch_i))
            
            gm_cfg.log_epoch_txt.flush()

        # end the train process(training_epoch_amount times to reuse the data)
        torch.save(obj = gm_net.state_dict(),  f = gm_cfg.name_save_model("ending"))
        gm_cfg.log_epoch_txt.close()
        gm_cfg.writer.close()

    except KeyboardInterrupt:
        print("Save the Inter.pth".center(60, "*"))
        torch.save(obj = gm_net.state_dict(), 
                    f = gm_cfg.name_save_model("interrupt"))


