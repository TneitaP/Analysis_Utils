import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim


import time 
import os 
import sys
ROOT = os.getcwd()
sys.path.append(ROOT)

from arch.Lenet5 import LeNet5_1998
import loader.basicLoader as bld
import custom_utils.config as ut_cfg 
import custom_utils.initializer as ut_init

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("please pip install tensorboard==2.0.2")


class train_config(ut_cfg.config):
    def __init__(self):
        super(train_config, self).__init__(pBs = 32, pWn = 2, p_force_cpu = False)
        self.path_save_mdroot = self.check_path_valid(os.path.join(ROOT, "outputs"))
        localtime = time.localtime(time.time())
        self.path_save_mdid = "convMNIST" + "%02d%02d"%(localtime.tm_mon, localtime.tm_mday)

        self.save_epoch_begin = 20
        self.save_epoch_interval = 10

        self.log_epoch_txt = open(os.path.join(self.path_save_mdroot, "conv_epoch_loss_log.txt"), 'a+')
        self.writer = SummaryWriter(log_dir=os.path.join(self.path_save_mdroot, "board"))

        self.height_in = 28
        self.width_in = 28
        self.latent_num = 16*5*5
        self.class_num = 10

        self.method_init = "xavier" #"preTrain" #"kaming"#"xavier"
        self.training_epoch_amount = 51

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
    
    def validate(self, pNet):
        # use the classifacation acc to validate the convNet performance
        valid_dataset = self.create_dataset(istrain=False)
        validloader = torch.utils.data.DataLoader(
            dataset = valid_dataset, 
            batch_size= self.ld_batchsize,
            shuffle= True,
            num_workers= self.ld_workers
        )

        acc_Lst = [] # len(validloader) = 313; 
        for iter_idx, (img_Tsor_bacth_i, label_Tsor_bacth_i) in enumerate(validloader):
            img_Tsor_bacth_i = img_Tsor_bacth_i.to(self.device)
            label_Tsor_bacth_i = label_Tsor_bacth_i.to(self.device)
            pred_Tsor_bacth_i = gm_net(img_Tsor_bacth_i)
            max_likeli_pred_bacth_i = torch.argmax(pred_Tsor_bacth_i,dim = -1)

            error_num = (max_likeli_pred_bacth_i - label_Tsor_bacth_i).nonzero().shape[0] 
            cur_acc = 1 - error_num / label_Tsor_bacth_i.shape[0]
            acc_Lst.append(cur_acc)
        
        return sum(acc_Lst) / len(acc_Lst)








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
    gm_net = LeNet5_1998(
            gm_cfg.width_in, gm_cfg.height_in, 
            gm_cfg.latent_num, gm_cfg.class_num)
    
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
    gm_criterion = nn.L1Loss()

    loss_an_epoch_Lst = []
    try:
        print("Train_Begin".center(40, "*"))
        gm_cfg.check_arch_para(gm_net)
        gm_cfg.log_in_file("net_id = ", gm_cfg.path_save_mdid, ", batchsize = ", gm_cfg.ld_batchsize, ", workers = ", gm_cfg.ld_workers)
        gm_cfg.log_in_file("criterion_use: ",gm_criterion, ", init: ", gm_cfg.method_init)
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
                label_Tsor_bacth_i = label_Tsor_bacth_i.to(gm_cfg.device)

                pred_Tsor_bacth_i = gm_net(img_Tsor_bacth_i)

                # create graph in tensorboard
                if iter_idx == 0:
                    gm_cfg.writer.add_graph(gm_net, img_Tsor_bacth_i)

                gm_optimizer.zero_grad() # clear old grad
                cur_loss = gm_criterion(pred_Tsor_bacth_i, label_Tsor_bacth_i)
                cur_loss.backward()  ## caculate new grad; retain_graph=True
                loss_an_epoch_Lst.append(cur_loss.item())

                gm_optimizer.step()   ### upgrade the para() using new grad
            # end an epoch
            delta_t = (time.time()- start)/60
            avg_loss = sum(loss_an_epoch_Lst)/len(loss_an_epoch_Lst)
            gm_scheduler.step(avg_loss)

            # validate the accuracy
            avg_acc = gm_cfg.validate(gm_net)

            gm_cfg.log_in_board("training loss", 
                {"avg_loss": avg_loss, 
                },  epoch_i
            )
            gm_cfg.log_in_board("validated acc", 
                {"avg_acc": avg_acc, 
                },  epoch_i
            )
            gm_cfg.log_in_file("epoch = %03d, time_cost(min)= %2.2f, loss = %2.5f, acc = %2.5f"
                %(epoch_i, delta_t, avg_loss, avg_acc)
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


    















