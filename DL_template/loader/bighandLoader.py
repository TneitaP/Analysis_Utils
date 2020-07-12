import os 
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import cv2 # for image read
from enum import IntEnum 


def view_tensor(p_img_Tsor):
    p_img_Tsor = p_img_Tsor    # unnormalize
    img_Arr = p_img_Tsor.numpy()
    plt.imshow(np.transpose(img_Arr, (1, 2, 0))) # C,H,W -> (H, W, C)
    plt.show()

class DataMode(IntEnum):
    TRACKING = 0
    HAND_OBJECT = 1

def img_load_helper(pImg_abs_dir):
    depth_Arr = cv2.imread(pImg_abs_dir, -1) # read as depth-map, uint16
    # to get better style for visualization and dtype to uint8
    qVisual_depth_Arr = cv2.convertScaleAbs(depth_Arr, alpha = 0.09) # uint8
    return qVisual_depth_Arr

class handtrace_Loader(torch.utils.data.Dataset):
    def __init__(self, pDataMode: DataMode, 
        pDataDir, pAnnofile_name, 
        pProject_Extrinsic: np.ndarray, pTansfom = None, 
        pData_helper = img_load_helper):
        '''
        pDataMode: TRACKING / HAND_OBJECT, two datasets in Hands2017 contest
        1) init the data dir and anno dir;
        2)  init the camera used in capturing the depth img
        '''

        self.data_dir = pDataDir
        # init the data dir and anno dir
        if pDataMode == DataMode.TRACKING:
            # r'E:\Database\Hands2017\tracking' , "mini_test_annotation_tracking.txt"
            # r'E:\HANDS2017\tracking' # server
            self.data_dir, self.track_sub_idx = pDataDir.split("__")
            self.anno_file_abs_dir = os.path.join(self.data_dir, pAnnofile_name)
            
        elif pDataMode == DataMode.HAND_OBJECT:
            # r'E:\Database\Hands2017\hand_object', "test_annotation_object.txt"
            self.data_dir = pDataDir
            track_sub_idx = ''
            self.anno_file_abs_dir = os.path.join(self.data_dir, pAnnofile_name)
        
        self.extrinsic = pProject_Extrinsic

        self.transform = pTansfom
        self.data_helper = pData_helper
        # open the anno-file and read all data-frame from it
        self.anoinfo_Lst = []
        with open(self.anno_file_abs_dir, 'r') as anno_file:
            for line_i in anno_file.readlines():
                info_i = line_i.split("\t   ")
                # get the cur image name
                if pDataMode == DataMode.TRACKING:
                    frame_id_i = info_i[0].split("\\")[-1] #  'tracking\\' + self.track_sub_idx + '\\images\\'+
                elif pDataMode == DataMode.HAND_OBJECT:
                    frame_id_i =  info_i[0].split("\\")[-1]

                # regularize the coord annotation
                ano_Lst_i = list(map(float, info_i[1:]))
                #jonts_Arr_i = np.array(pAno_Lst_i).reshape((-1,3)) # 21, 3
                # trans from world coord to camera coord
                #jonts_Arr_i = np.matmul(self.camera.extrinsic[:3,:3], jonts_Arr_i.T).T
                
                self.anoinfo_Lst.append([frame_id_i, ano_Lst_i])
                # frame_id_i is the img name, ano_Lst_i is the list formart of 3D coord

    def __len__(self):
        return len(self.anoinfo_Lst)

    def __getitem__(self, index):
        frame_id_i, ano_Lst_i = self.anoinfo_Lst[index]

        # get image(gray-scale):
        frame_abs_dir = os.path.join(self.data_dir, self.track_sub_idx, "images", frame_id_i)
        assert os.path.isfile(frame_abs_dir), "wrong image abspath in <handtrace_Loader.getitem>, %s"%(frame_abs_dir)
        depth_Arr = self.data_helper(frame_abs_dir)
        if not self.transform:
            depth_Tsor = torch.Tensor(depth_Arr)
        else:
            depth_Tsor = self.transform(depth_Arr)
        if depth_Tsor.dim() == 2:
            # image : (N, C=1 , H, W)
            depth_Tsor = depth_Tsor.unsqueeze(0)
        
        # get anno coord 3D
        jonts_Arr = np.array(ano_Lst_i).reshape((-1,3)) # 21, 3
        # trans from world coord to camera coord
        jonts_Arr = np.matmul(self.extrinsic[:3,:3], jonts_Arr.T)

        if not self.transform:
            jonts_Tsor = torch.Tensor(jonts_Arr)
        else:
            jonts_Tsor = self.transform(jonts_Arr)

        # torch.Tensor
        return frame_id_i, depth_Tsor,  jonts_Tsor



if __name__ == "__main__":
    gm_extrinsic = np.array([
                    [1,  0, 0, 0],
                    [0, -1, 0, 0], 
                    [0, 0, -1, 0], 
                    [0, 0,  0, 1]
                    ])

    big_data_dir = r'E:\Database\Hands2017\tracking_sample'+'__2'
    annofile_name = "sub_test_annotation_tracking_2_2781.txt"  

    gm_dataset = handtrace_Loader(DataMode.TRACKING, big_data_dir, annofile_name, gm_extrinsic)
    trainloader = torch.utils.data.DataLoader(dataset = gm_dataset, batch_size = 16, 
                        shuffle= False, num_workers = 2)
                        
    for img_id_bacth_i, depth_Tsor_bacth_i, jonts_Tsor_bacth_i in trainloader:
        print(len(img_id_bacth_i), depth_Tsor_bacth_i.shape, jonts_Tsor_bacth_i.shape)
        view_tensor(
            torchvision.utils.make_grid(
                        tensor = depth_Tsor_bacth_i, 
                        nrow= 4)
                    )


