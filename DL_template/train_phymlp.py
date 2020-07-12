import time 
import random
import os 
import sys
ROOT = os.getcwd()
sys.path.append(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("please pip install tensorboard==2.0.2")


import custom_utils.config as ut_cfg 
import custom_utils.initializer as ut_init
import loader.phyLoader as phyld

from arch.phy_mlp import qpos2force



class train_config(ut_cfg.config):
    def __init__(self):
        super(train_config, self).__init__(pBs = 2, pWn = 2, p_force_cpu = False)
        self.path_save_mdroot = self.check_path_valid(os.path.join(ROOT, "outputs"))
        localtime = time.localtime(time.time())
        self.path_save_mdid = "qpos2force" + "%02d%02d"%(localtime.tm_mon, localtime.tm_mday)
        self.save_epoch_begin = 9
        self.save_epoch_interval = 10

        self.log_epoch_txt = open(os.path.join(self.path_save_mdroot, "convVae_epoch_loss_log.txt"), 'a+')
        self.writer = SummaryWriter(log_dir=os.path.join(self.path_save_mdroot, "board"))

        self.method_init ="norm"  #"preTrain" #"kaming" #"xavier" # "norm"
        self.training_epoch_amount = 100

        self.dtroot = os.path.join(r"F:\ZimengZhao_Data\phy_hand_cash")
        self.file_amount = 2048

        self.opt_baseLr = 5e-4
        self.opt_bata1 = 0.5
        self.opt_weightdecay = 3e-6
    
    def init_net(self, pNet):
        if self.method_init == "xavier":
            ut_init.init_xavier(pNet)
        elif self.method_init == "kaiming":
            ut_init.init_kaiming(pNet)
        elif self.method_init == "norm":
            ut_init.init_norm(pNet)
        elif self.method_init == "preTrain":
            assert self.preTrain_model_path is not None, "weight path ungiven"
            pNet.load_state_dict(torch.load(self.preTrain_model_path))

        pNet.to(self.device).train()
    
    def create_dataset(self, istrain):
        q_dataset = phyld.forcepos_Loader(self.dtroot, self.file_amount)
        return q_dataset
    
    def name_save_model(self, save_mode, epochX = None):
        model_type = save_mode.split("_")[1] # netD / netG
        model_filename = self.path_save_mdid + model_type
        
        if "processing" in save_mode:
            assert epochX is not None, "miss the epoch info" 
            model_filename += "_%03d"%(epochX) + ".pth"
        elif "ending" in save_mode:
            model_filename += "_%03d"%(self.training_epoch_amount) + ".pth"
        elif "interrupt" in save_mode:
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

if __name__ == "__main__":
    

    gm_cfg = train_config()

    # prepare data
    gm_trainloader = torch.utils.data.DataLoader(
        dataset = gm_cfg.create_dataset(istrain = True), 
        batch_size= gm_cfg.ld_batchsize,
        shuffle= True,
        num_workers= gm_cfg.ld_workers
    ) # 

    gm_net = qpos2force()
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

    # loss criterion
    gm_criterion = nn.CrossEntropyLoss()

    loss_an_epoch_Lst = []
    try:
        print("Train_Begin".center(40, "*"))
        gm_cfg.check_arch_para(gm_net)
        gm_cfg.log_in_file("net_id = ", gm_cfg.path_save_mdid, ", batchsize = ", gm_cfg.ld_batchsize, ", workers = ", gm_cfg.ld_workers)
        gm_cfg.log_in_file("criterion_use: ",gm_criterion, ", init: ", gm_cfg.method_init)
        for epoch_i in range(gm_cfg.training_epoch_amount):
            start=time.time()
            # single epoch
            for iter_idx, datapac_i in enumerate(gm_trainloader):
                gt_force_Tsor_batch_i, qpos_Tsor_batch_i, xpos_Tsor_batch_i = datapac_i

                gt_force_Tsor_batch_i = gt_force_Tsor_batch_i.to(gm_cfg.device).view(-1, 24)
                qpos_Tsor_batch_i = qpos_Tsor_batch_i.to(gm_cfg.device).view(-1, 24)

                pred_force_Tsor_batch_i = gm_net(qpos_Tsor_batch_i)

                gm_optimizer.zero_grad() # clear old grad
                cur_loss = gm_criterion(pred_force_Tsor_batch_i, gt_force_Tsor_batch_i)
                cur_loss.backward()  ## caculate new grad; retain_graph=True
                loss_an_epoch_Lst.append(cur_loss.item())

                gm_optimizer.step()   ### upgrade the para() using new grad
            # end an epoch
            delta_t = (time.time()- start)/60
            avg_loss = sum(loss_an_epoch_Lst)/len(loss_an_epoch_Lst)
            gm_scheduler.step(avg_loss)

            # log here. 
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