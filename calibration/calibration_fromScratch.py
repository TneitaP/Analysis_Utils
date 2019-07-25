import os
import cv2
import numpy as np



if __name__ == "__main__":



    ##################################################################################
    ##########################Step1. Input Image List and Get the Point Coordinate####
    ##########################  in both 2Dimg and 3Dobj ##############################
    ##################################################################################
    # C:\lib\opencv_selfbuild\opencv-3.4.2\samples\data
    # 存放图像的位置：
    CAMERA_CHESSBOARD_PATH = r"C:\Users\TneitaP\Documents\Visual Studio 2015\Projects\openCV_calibration\x64\Debug\example"
    
    CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # final objp shape = (42, 3); 每行 (0,0,0) -> ...->(6,5,0);
    t_W = 6
    t_H = 7
    objp = np.zeros((t_W*t_H,3), np.float32) 
    objp[:,:2] = np.mgrid[0:t_H,0:t_W].T.reshape(-1,2)

    Gimages_Lst = []
    # Arrays to store object points and image points from all the images. 
    objpoints = [] # 3d point in real world space 对象点
    imgpoints = [] # 2d points in image plane.图像点
    t_imgW = t_imgH = 0 # 反序读取 .shape
    for fileName_i in os.listdir(CAMERA_CHESSBOARD_PATH):
        if "left0" not in fileName_i \
            and "left1" not in fileName_i \
            or os.path.splitext(fileName_i)[1] != ".jpg":
            continue
        # needed img
        print(os.path.join(CAMERA_CHESSBOARD_PATH, fileName_i)) # 01-14
        cur_img = cv2.imread(os.path.join(CAMERA_CHESSBOARD_PATH, fileName_i))
        # print(cur_img.shape) #(480(y_max= Row), 640(x_max= Col), 3)
        cur_Gimg = cv2.cvtColor(cur_img,cv2.COLOR_BGR2GRAY) # to grey scale

        if t_imgW == 0 and t_imgH == 0:
            t_imgW, t_imgH = cur_Gimg.shape[::-1] # [::-1]代表反序读取

        ret, corners = cv2.findChessboardCorners(cur_Gimg, (7,6),None)
        
        # print(cur_img.shape) (480, 640)
        Gimages_Lst.append(cur_Gimg)
        # print("Added ~",len(images_Lst))
        if ret == True: 
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(cur_Gimg,corners,(11,11),(-1,-1),CORNER_CRITERIA)
            imgpoints.append(corners2)

            #cv2.drawChessboardCorners(cur_img, (7,6), corners2,ret) 
            #cv2.imshow('Chess Board Corner' + fileName_i,cur_img)
            #cv2.waitKey(0)
    print("[Valid Count]Final Objpoints Lst", len(objpoints))
    print("[Valid Count]Final Imgpoints Lst", len(imgpoints))
    #cv2.destroyAllWindows()
    
    ##################################################################################
    ##########################Step2. Analyze the Cemera Model use ####################
    ################################   Objpoints && Imgpoints ########################
    ##################################################################################
    RMS, mtx, dist, rvecs_Lst, tvecs_Lst = \
        cv2.calibrateCamera(objpoints, imgpoints, (t_imgW, t_imgH),None,None)
    # select_pice = 3
    # for i in range(42):
    #     print("i = %d"%(i))
    #     ret, rvecs_single, tvecs_sigle, inliers_single = cv2.solvePnPRansac(objpoints[i], imgpoints[i], mtx, dist)
    #     print("R:")
    #     print(rvecs_single,"\n",  rvecs_Lst[i])
    #     print("T:")
    #     print(tvecs_sigle,"\n", tvecs_Lst[i])
    # <float>ret,   平均重新投影误差 RMS【良好校准时应在0.1和1.0像素之间】
    # <numpy.ndarray>mtx, 内参矩阵 camera matrix
    # <numpy.ndarray>dist , 畸变系数 Distortion coefficients
    # <Lst> rvecs_Lst 外参矩阵的 旋转参数
    # <list>tvecs_Lst 外参矩阵的 平移参数
    # print("Test the Unknow ret: \n", ret)
    # print("Test the Unknow mtx: \n", mtx)
    # print("Test the Unknow dist: \n", dist)
    # print("Test the Unknow rvecs_Lst: \n", type(rvecs_Lst))
    # print("Test the Unknow tvecs_Lst: \n", type(tvecs_Lst))
    # for i,R_i in enumerate(rvecs_Lst) :
    #     print("R in img_(%d) \n"%(i), R_i)
    

    # for i,T_i in enumerate(tvecs_Lst) :
    #     print(" T in img_(%d) \n"%(i), T_i)


    ##################################################################################
    ########################## Step3. Remap the Raw img ##############################
    ##################################################################################

    for Gimg_i in Gimages_Lst:
        # 获取新的投影矩阵，入参：
        # (cameraMatrix, distCoeffs, imageSize, alpha, newImgSize, centerPrincipalPoint)
        # 如果缩放系数 alpha = 0，返回的非畸变图像会带有最少量的不想要的像素
        # 如果 alpha = 1，所 有的像素都会被返回，还有一些黑图像。
        # 它还会返回一个 ROI 图像，我们可以 用来对结果进行裁剪。
        # (t_imgW,t_imgH) 只影响 roi 计算， 不影响 新矩阵获取
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(t_imgW,t_imgH),1,(t_imgW,t_imgH))
        print(roi)
        # way 1). 裁剪，入参：内参 + 畸变
        #(src, cameraMatrix, distCoeffs, dst, newCameraMatrix)
        dst = cv2.undistort(Gimg_i, mtx, dist, None, newcameramtx)
        # way 2). 重映射：
        mapx,mapy = cv2.initUndistortRectifyMap(mtx, dist, None,
            newcameramtx,(t_imgW,t_imgH),5)
        dst = cv2.remap(Gimg_i,mapx,mapy,cv2.INTER_LINEAR)
        print(mapx,mapy)
        x,y,w,h = roi 
        dst = dst[y:y+h, x:x+w]

        ### 对比前后:
        result_compare = np.zeros((t_imgH, t_imgW*2), np.uint8)
        print("Cur Scale: ", t_imgH, t_imgW)
        print("result_compare Shape = ", result_compare[0:t_imgH, 0:t_imgW].shape)
        result_compare[0:t_imgH, 0:t_imgW] = Gimg_i

        # 尺寸 padding (top, bottom, left, right)
        dst= cv2.copyMakeBorder(dst,t_imgH - dst.shape[0],0,t_imgW - dst.shape[1],0,cv2.BORDER_CONSTANT,value=(0,0,0))
        result_compare[0:t_imgH, t_imgW:t_imgW*2] = dst
        cv2.imshow("cur Remap img:", result_compare)
        cv2.waitKey(0)
        
    
    cv2.destroyAllWindows()