import numpy as np 
import torch 
import os 


# def btchforce_load_helper(pForce_np_dir):
#     force_Arr = np.load(pForce_np_dir) # (4096, 24)
#     return force_Arr

def np_load_helper(npArr_dir):
    phy_Arr = np.load(npArr_dir) # (4096, 24)
    return phy_Arr.astype(np.float32)

class forcepos_Loader(torch.utils.data.Dataset):

    def check_file(self, proot, pfilename):
        file_fullname = os.path.join(proot, pfilename)
        if os.path.isfile(file_fullname):
            return True
        else:
            print("abs filename does not exist. "+ pfilename)
            return False

    def __init__(self, pDataRoot, pFile_amount, pTansfom = None, pData_helper = np_load_helper):
        
        if not pTansfom:
            self.transform = torch.from_numpy #torch.Tensor()
        else:
            self.transform = pTansfom
            
        self.data_helper = pData_helper

        assert os.path.isdir(pDataRoot), "invalid data root: " + pDataRoot
        self.force_dir_Lst = []
        self.qpos_dir_Lst = []
        self.xpos_dir_Lst = []
        
        for data_idx in range(pFile_amount):
            frc_fi = "force_seq_%05d.npy"%(data_idx)
            qpos_fi = "qpos_seq_%05d.npy"%(data_idx)
            xpos_fi = "xpos_seq_%05d.npy"%(data_idx)
            if self.check_file(pDataRoot, frc_fi) and self.check_file(pDataRoot, qpos_fi) and self.check_file(pDataRoot, xpos_fi):
                self.force_dir_Lst.append(os.path.join(pDataRoot, frc_fi))
                self.qpos_dir_Lst.append(os.path.join(pDataRoot, qpos_fi))
                self.xpos_dir_Lst.append(os.path.join(pDataRoot, xpos_fi))
        
    def __len__(self):
        return len(self.force_dir_Lst)
    
    def __getitem__(self, index):
        force_absdir = self.force_dir_Lst[index]
        qpos_absdir = self.qpos_dir_Lst[index]
        xpos_absdir = self.xpos_dir_Lst[index]

        force_Arr = self.data_helper(force_absdir)
        qpos_Arr = self.data_helper(qpos_absdir)
        xpos_Arr = self.data_helper(xpos_absdir)

        # if not self.transform:
        #     force_Tsor = torch.Tensor(force_Arr)
        #     qpos_Tsor = torch.Tensor(qpos_Arr)
        #     xpos_Tsor = torch.Tensor(xpos_Arr)

        force_Tsor = self.transform(force_Arr)
        qpos_Tsor = self.transform(qpos_Arr)
        xpos_Tsor = self.transform(xpos_Arr)

        return force_Tsor, qpos_Tsor, xpos_Tsor
        

if __name__ == "__main__":
    # test loader

    gm_dataset = forcepos_Loader(pDataRoot=r"F:\ZimengZhao_Data\phy_hand_cash", pFile_amount = 2048)

    trainloader = torch.utils.data.DataLoader(dataset = gm_dataset, batch_size = 16, 
                        shuffle= False, num_workers = 2)
    
    for datapac_i in trainloader:
        force_Tsor_batch_i,  qpos_Tsor_batch_i, xpos_Tsor_batch_i = datapac_i

        print("chek force shape: ", force_Tsor_batch_i.shape)
        print("chek qpos shape: ", qpos_Tsor_batch_i.shape)
        print("chek xpos shape: ", xpos_Tsor_batch_i.shape)
