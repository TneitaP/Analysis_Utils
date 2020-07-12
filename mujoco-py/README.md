**Status:** Maintenance (expect bug fixes and minor updates)

# mujoco-py [![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://openai.github.io/mujoco-py/build/html/index.html) [![Build Status](https://travis-ci.org/openai/mujoco-py.svg?branch=master)](https://travis-ci.org/openai/mujoco-py)

[MuJoCo](http://mujoco.org/) is a physics engine for detailed, efficient rigid body simulations with contacts.
![image](illus/mujoco.png)
`mujoco-py` allows using MuJoCo from Python 3. This library has been updated to be compatible with MuJoCo version 2.0 released on 10/1/2018.

It is used in many well-known laboratories current research, such as:
- [gym](https://gym.openai.com/envs/#mujoco) RL platform created by OpenAI; 
- [ppdm](https://github.com/google-research/pddm) a In-hand manipulations algrithom created by Google Brain; 
- DeepMind's [dm_control](https://github.com/deepmind/dm_control/tree/master/dm_control/locomotion) and [rodent control](https://openreview.net/forum?id=SyxrxR4KPS);
- [Mujoco Haptix](http://www.mujoco.org/book/haptix.html) , a glove-based system only runs on Windows, mujoco150 up to now. 


## Windows Adaption
Following the [issues#504](https://github.com/openai/mujoco-py/issues/504), this repo adapt the latest version of mujoco python wrapper to the windows environment correctly. 
The following demo is the gym env `env = gym.make('HandManipulatePenTouchSensorsDense-v0')`. 
![image](illus/hand.png)

### Pre-Conditions
One should firstly apply a valid mujoco license in [the license page](https://www.roboti.us/license.html) and received a file called `mjkey.txt`.Then you could download this C/C++ API in [the index page](https://www.roboti.us/index.html). 
After all these works, you could navigate to `mujoco\mujoco200\bin` directory with the `mjkey.txt` copied in, and run the commands like： 

01. **simulate 对xml定义的场景进行仿真【被动】** Usage: use glfw to view the results:
```
simulate ../model/humanoid.xml
simulate ../model/hello.xml
```


02. **compile 模型转换** Usage: compile infile outfile
```
infile can be in mjcf, urdf, mjb format
outfile can be in mjcf, mjb, txt format
```

This is the base usage of Mujoco. For dive-into demo, please check the `\sample` directory.
 
### Environment Variable Settings

Here is my specific settings table. 

| Name |                    Variable | Description | 
|------|-----------------------------|-----------|
| MUJOCO_PY_MUJOCO_PATH | D:\Libraries\mujoco\mujoco200 | The dir contain mjc's bin, include, etc. |
|MUJOCO_PY_MJKEY_PATH | D:\Libraries\mujoco\utils\mjkey.txt | The abs path of the mjkey | 
| PATH | += D:\Libraries\mujoco\mujoco200\bin | needed by cl.exe; for lib and dll | 
|INCLUDE | D:\Libraries\mujoco\mujoco200\include | needed by cl.exe;for header files |  



### Python Wrapper Creation
After the two above-mentioned Settings, you can create a anaconda environment with these configurations: 
- Windows 10
- Anaconda Python 3.7.6
- Mujoco200

And what will be created is: 
- mujoco-py 2.0.2.9



## Credits

`mujoco-py` is maintained by the OpenAI Robotics team. Contributors include:

- Alex Ray
- Bob McGrew
- Jonas Schneider
- Jonathan Ho
- Peter Welinder
- Wojciech Zaremba
- Jerry Tworek
