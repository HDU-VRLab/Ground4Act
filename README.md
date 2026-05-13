# Ground4Act: Leveraging Visual-Language Model for Collaborative Pushing and Grasping in Clutter

🌟This repository contains the implementation of [Ground4Act](https://www.sciencedirect.com/science/article/pii/S0262885624003858), a two-stage approach for collaborative pushing and grasping in clutter using a visual-language model.📗[**Demonstration**](#demonstration) | [**Installation**](#installation) | [**Model Weights**](#model-weights) | [**Getting Started**](#getting-started) | [**Related Work**](#related-work) | [**BibTeX**](#bibtex)

## Demonstration

🤖[Video](https://www.bilibili.com/video/BV1Jj2nY1EhR/?spm_id_from=333.999.0.0&vd_source=63cd8055657905c0ac8a9388d7a972ed)
🌐[Personal homepage](https://space.bilibili.com/485363351/video)

## Installation

The repository is based on ubuntu18.04.

Before you start, ensure that [ROS (Robot Operating System)](http://wiki.ros.org/) is installed on your system.

### Step 1: Clone the Repository

Open your terminal (**Python 2**) and run the following command to clone the repository:

```bash
mkdir ur_ws && cd ur_ws
git clone https://github.com/HDU-VRLab/Ground4Act.git
```

Install the necessary libraries under the current terminal for [push network](https://github.com/nizhihao/Collaborative-Pushing-Grasping).

```bash
sudo chmod +x install_ros_packages.sh
./install_ros_packages.sh
catkin_make
pip install torch==1.0.0 scipy==1.2.3 torchvision==0.2.1
```

### Step 2: Create a new environment

Create a **Python 3** virtual environment using [conda](https://docs.conda.io/en/latest/). For information on Visual Grounding, please refer to [RefTR](https://github.com/ubc-vision/RefTR).

```bash
conda create -n Vlpg python=3.7
conda activate Vlpg
pip3 install torch torchvision torchaudio scikit-image
cd vl_grasp/RoboRefIt
pip3 install -r requirements.txt 
```

## Model Weights

| Resource             | Description          |
|----------------------|----------------------|
| [Sim_model](https://github.com/nizhihao/Collaborative-Pushing-Grasping/tree/master/myur_ws/src/ur_robotiq/ur_robotiq_gazebo/meshes) | Place the downloaded simulation model under "/home/xxx/.gazebo/models". |
| [Ground4Act](https://pan.baidu.com/s/1jalj3nmUaaE2AAztAjAgfw?pwd=1234) |Place the downloaded Push network weight in "src\gjt_ur_moveit_gazebo\env_info\push.pth".<br> The Visual Grounding weight is placed in "src\vl_grasp\logs". |

## Getting Started

### Usage Guidelines

When using ROS with MoveIt for control, please follow these guidelines:

- The terminal for controlling the system must run **Python 2**.
- The terminal for executing algorithm must run **Python 3**.

This setup is crucial for ensuring proper functionality and compatibility between the different components of the system.

### Step 1: Launch simulation and load MoveIt

Please turn on the simulation button in the lower left corner of gazebo, then you can control the robot through [MoveIt](https://moveit.ros.org/).

```bash
cd ur_ws
source ./devel/setup.bash
roslaunch gjt_ur_moveit_gazebo start_gjt_ur_moveit_gazebo.launch 
```

We provide many useful [unit test scripts](src/gjt_ur_moveit_gazebo/gazebo_scripts). Preloading the object model in gazebo helps with later execution speed. 

So it is recommended to run in sequence at terminal 2:

```bash
python src/gjt_ur_moveit_gazebo/gazebo_scripts/spawn_model.py
python src/gjt_ur_moveit_gazebo/gazebo_scripts/moveitServer.py
```

### Step 2: Executing algorithm

```bash
conda activate Vlpg
python src/vl_grasp/vl_push_grasp.py
```

## Related Work

Many thanks to previous researchers for sharing their excellent work:

```bibtex
@article{yang2021collaborative,
  title={Collaborative pushing and grasping of tightly stacked objects via deep reinforcement learning},
  author={Yang, Yuxiang and Ni, Zhihao and Gao, Mingyu and Zhang, Jing and Tao, Dacheng},
  journal={IEEE/CAA Journal of Automatica Sinica},
  volume={9},
  number={1},
  pages={135--145},
  year={2021},
  publisher={IEEE}
}

@inproceedings{lu2023vl,
  title={VL-Grasp: a 6-Dof Interactive Grasp Policy for Language-Oriented Objects in Cluttered Indoor Scenes},
  author={Lu, Yuhao and Fan, Yixuan and Deng, Beixing and Liu, Fangfu and Li, Yali and Wang, Shengjin},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={976--983},
  year={2023},
  organization={IEEE}
}

@inproceedings{muchen2021referring,
  title={Referring Transformer: A One-step Approach to Multi-task Visual Grounding},
  author={Muchen, Li and Leonid, Sigal},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```

## BibTeX

If you find our code or models useful in your work, please cite [our paper](https://www.sciencedirect.com/science/article/pii/S0262885624003858).

```bibtex
@article{YANG2024105280,
  title = {Ground4Act: Leveraging visual-language model for collaborative pushing and grasping in clutter},
  author = {Yuxiang Yang and Jiangtao Guo and Zilong Li and Zhiwei He and Jing Zhang},
  journal = {Image and Vision Computing},
  volume = {151},
  pages = {105280},
  year = {2024},
  url = {https://www.sciencedirect.com/science/article/pii/S0262885624003858}
}
```


## Deploy to AgileX Piper (ROS Noetic)

> 当前环境无法直接联网拉取 `piper_ros`（GitHub 403），所以我把“可直接替换/运行”的落地方式写成了**可执行步骤**。

### 1) 在你的 Piper 工作空间中放置 Ground4Act 代码

假设你的 Piper 工作空间是 `~/piper_ws`，把本仓库放到同一个 `src` 下：

```bash
cd ~/piper_ws/src
# 放入 Ground4Act 项目目录（例如 git clone 或拷贝）
```

### 2) 直接使用一键启动文件（不再手动 rosparam load）

我已新增：
- `src/gjt_ur_moveit_gazebo/launch/start_piper_ground4act.launch`

它会自动：
1. 加载 `piper_adapter_params.yaml`
2. 启动 `moveitServer.py`
3. 暴露 `moveit_grasp`（可改名）服务

启动命令：

```bash
roslaunch gjt_ur_moveit_gazebo start_piper_ground4act.launch
```

### 3) 你只需要改一个文件

按你的 Piper MoveIt 配置修改：
- `src/gjt_ur_moveit_gazebo/config/piper_adapter_params.yaml`

重点字段：
- `arm_group_name`
- `home_joint_values`
- `gripper_action_ns`
- `gripper_joint_name`
- `work_surface_z` / offsets

### 4) 与 piper_ros 的衔接方式

先启动 Piper 官方 MoveIt（noetic）后，再启动 Ground4Act 服务。流程如下：

1. 启动 piper_ros 的机器人驱动 + MoveIt。
2. 执行：`roslaunch gjt_ur_moveit_gazebo start_piper_ground4act.launch`。
3. 复用原 `moveit_grasp` 服务调用链（push/grasp/reset）。

### 5) 如果你要“直接替换到 piper_ros 包内”

不建议改官方包源码，推荐通过工作空间叠加。
如果你一定要替换，可将本项目 `gjt_ur_moveit_gazebo/gazebo_scripts` 下脚本复制到你自己的控制包，并保留同名服务接口 `moveit_grasp`。
