Calibration based on opncv;2019/07/24
详细说明见\\x64\\ 的Log.txt;
通过python调用.exe的脚本见\\x64\\upperOperate.py；
测试数据包含opencv 自带的左右，本人自己采集的左右。

整个过程分为两步，
首先由imagelist_creator制作yaml文件记录图像位置；
之后通过calibration逐个图计算内参矩阵；