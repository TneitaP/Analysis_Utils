# Calibration based on opncv;2019/07/24

1. 调用C++ API（本人使用debug X64 模式）
详细说明见\\x64\\ 的Log.txt;
通过python调用.exe的脚本见\\x64\\upperOperate.py；
测试数据包含opencv 自带的左右，本人自己采集的左右。

整个过程分为两步，
首先由imagelist_creator制作yaml文件记录图像位置；<imagelist_creator.cpp>编译生成exe
之后通过calibration逐个图计算内参矩阵；<calibration.cpp>编译生成exe

<calibration_UsingEXE.py>实现的是自动调用上述编译好的exe, 生成对应的yaml;

2. python API From Scratch
<calibration_fromScratch.py>实现；
