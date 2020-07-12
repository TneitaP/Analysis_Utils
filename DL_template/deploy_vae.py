from PyQt5 import QtGui, QtCore, QtWidgets
import sys
import numpy as np
import cv2

import torch
import torch.nn as nn
# from arch.vae import convVAE
from arch.vae import fcVAE
import custom_utils.config as ut_cfg 

import time 
import os 
import sys
ROOT = os.getcwd()
sys.path.append(ROOT)

class test_config(ut_cfg.config):
    def __init__(self):
        super(test_config, self).__init__(pBs = 32, pWn = 2, p_force_cpu = False)
        self.path_save_mdroot = self.check_path_valid(os.path.join(ROOT, "outputs_vae"))

        self.path_save_mdid = "fcVaeMNIST0326_500.pth"

        self.method_init = "preTrain"

        self.height_in = 28
        self.width_in = 28
        self.latent_num = 16

    def init_net(self, pNet):
        assert self.method_init == "preTrain"
        pretrained_weight_path = os.path.join(self.path_save_mdroot, self.path_save_mdid)
        print("loading...", pretrained_weight_path)
        pNet.load_state_dict(torch.load(pretrained_weight_path))

        pNet.to(self.device).eval()
    

class WindowClass(QtWidgets.QWidget):
    
    def __init__(self,parent=None):
        super(WindowClass, self).__init__(parent)
        m_layout=QtWidgets.QGridLayout()
        self.prepareModel()

        # allocate the elements
        self.slider_num = 16
        
        self.slider_Lst = []
        self.label_Lst = []
        
        for i in range(self.slider_num):
            # set slider
            cur_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal) # h slider
            # cur_slider.objectName
            # cur_slider.setValue(0)
            cur_slider.setObjectName("slider_"+ str(i))
            cur_slider.setMinimum(0)#最小值
            cur_slider.setMaximum(100)#最大值
            cur_slider.setSingleStep(1)#步长
            cur_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)#设置刻度位置，在下方
            cur_slider.setTickInterval(5)#设置刻度间隔
            
            m_layout.addWidget(cur_slider, i+1, 1) # (0,0) is window
            self.slider_Lst.append(cur_slider)
            
            # set label
            cur_label = QtWidgets.QLabel()
            cur_label.setObjectName("figure_"+ str(i))
            cur_label.setFont(QtGui.QFont(None,10))
            cur_label.setText('000')
            
            m_layout.addWidget(cur_label, i+1, 0) # (0,0) is window
            self.label_Lst.append(cur_label)

        for i in range(self.slider_num):
            self.slider_Lst[i].valueChanged.connect(self.valChange)
        
        # 03. init the img
        self.image_frame = QtWidgets.QLabel()

        self.image_Arr = None 
        self.updateImage()
        
        self.image_qt = QtGui.QImage(self.image_Arr.data, self.image_Arr.shape[1], self.image_Arr.shape[0], QtGui.QImage.Format_Grayscale8)
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image_qt))
        m_layout.addWidget(self.image_frame, 1,2)

        self.resize(500,500)
        self.setLayout(m_layout)
        

    def prepareModel(self):
        self.cfg = test_config()
        self.vae_net = fcVAE(self.cfg.latent_num)
        self.cfg.init_net(self.vae_net)
    
    def updateImage(self):
        # get current Z:
        z_Tsor = torch.rand(16)
        # z_Tsor = torch.tensor([0.9655, -0.5851, -0.6322,  0.5465, -1.7639, -0.8880, -1.1397, -0.3770,
        #     1.5458, -1.3838,  0.8842, -0.9229,  1.3078, -0.8366, -0.1422,  0.7480])
        for i in range(len(z_Tsor)):
            z_Tsor[i] = self.slider_Lst[i].value() / 50.0 - 1 # [0, 100] -> [0, 2] -> [-1, 1]
        z_Tsor = z_Tsor.unsqueeze(0)
        rec_Tsor = self.vae_net.decode(z_Tsor.to(self.cfg.device))
        if rec_Tsor.is_cuda:
            rec_Tsor = rec_Tsor.cpu()
        if rec_Tsor.requires_grad:
            rec_Tsor = rec_Tsor.detach()

        rec_Arr = rec_Tsor.view(28,28).numpy()
        rec_Arr = (255*rec_Arr).astype(np.uint8)
        rec_Arr = cv2.resize(rec_Arr,  (160, 160))
        self.image_Arr = rec_Arr





    def valChange(self, index:int):

        for i in range(self.slider_num):
            self.label_Lst[i].setText(str(self.slider_Lst[i].value()).zfill(3))

        self.updateImage()
        self.image_qt = QtGui.QImage(self.image_Arr.data, self.image_Arr.shape[1], self.image_Arr.shape[0], QtGui.QImage.Format_Grayscale8)
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image_qt))

if __name__=="__main__":
    app=QtWidgets. QApplication(sys.argv)
    win=WindowClass()
    win.show()
    sys.exit(app.exec_())

