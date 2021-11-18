# map_evaluation_tools
This tool computes the *Mean Map Entropy* and the *Mean Plane Variance* of a point cloud.

##Dependencies
- PCL
- OpenMP
- Eigen
- [pybind11](https://github.com/pybind/pybind11)

## Install
1. First, clone this project，notice the SUBMODULE should also be cloned.
````bash
git clone https://github.com/hyxhope/map_evaluation_tools.git --recursive
````

2. Compile and install.

```bash
python setup.py install
```

## Usage
For details, please refer scripts/example.py

###Example

````python
import numpy as np
import map_evaluation

# 从pcd文件加载点云
pcd_file = "your/path/to/pointcloud.pcd"
map_evaluation.init(pcd_file=pcd_file, voxelLeaf=0.3, minKNearest=5, stepSize=2, radius=1.0)

# map_evaluation同时支持numpy传入N*4点云
pc = np.empty((25000, 4)) # x,y,z,intensity
map_evaluation.init(input=pc, voxelLeaf=0.3, minKNearest=5, stepSize=2, radius=1.0)

# mme, mpv
print(map_evaluation.mme())
print(map_evaluation.mpv())
````

###Parameter
````yaml
// 参数设置(Optional)
float voxelLeaf;    // 降采样阈值(可以为0，表示不进行降采样操作)
int minKNearest;    // 最小最近邻邻居的个数
int stepSize;       // 步长(每隔几个点计算一次，可加快计算速度)
float radius;       // 最近邻搜索阈值
````

##TODO
Add mom