

import os


def Cut_pathString(pLong:str, pShort:str):
    cache_index = 0
    for index, (char_i, char_j) in enumerate(zip(pLong, pShort)):
        if char_i!=char_j:
            break
        else:
            cache_index = index
    return pLong[index+2:]


gEXEwork_PATH = r"C:\Users\TneitaP\Documents\Visual Studio 2015\Projects\openCV_calibration\x64\Debug"
gCAMERA_CHESSBOARD_PATH = r"C:\Users\TneitaP\Documents\Visual Studio 2015\Projects\openCV_calibration\x64\Debug\self\img"

gKeyWord = "R"
gYAML_nameLst = gKeyWord + "_samples_imglist.yaml"
gImgType = ".png"

gW = 5
gH = 7
if __name__ == "__main__":
    
    os.chdir(gEXEwork_PATH)

    t_cmd_part1 = "part1_imagelist " + gYAML_nameLst + " "
    t_cmd_part2 = "part2_calibration "+ gYAML_nameLst + " "

    t_img_Lst = ""
    for i, fileName_i in enumerate( os.listdir(gCAMERA_CHESSBOARD_PATH) ):
        if gKeyWord not in fileName_i or os.path.splitext(fileName_i)[1] != gImgType:
            continue
        # Left camera
        # print(fileName_i)
        t_img_Lst += (" " + os.path.join(Cut_pathString(gCAMERA_CHESSBOARD_PATH, gEXEwork_PATH), fileName_i) ) #
        # if i%4 == 0:
        #     t_cmd_part1 += " ^ \n"
    
    print("[info]creating nameList yaml..")
    t_cmd_part1 += t_img_Lst
    print(t_cmd_part1)
    part1_return = os.system(t_cmd_part1)
    assert part1_return == 0, "[Error]wrong in 1st Step."
    
    print("[info]counting intrinsic matrix..")
    t_cmd_part2 += t_img_Lst
    t_cmd_part2 += " -w=%d -h=%d -o=%s"%(gW, gH, gKeyWord + "_out_camera_data.yml")
    part2_return = os.system(t_cmd_part2)
    #print(part2_return)


