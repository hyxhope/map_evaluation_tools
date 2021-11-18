//
// Created by hyx on 2021/11/18.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "map_evaluation.hpp"

map_evaluation ME;

void init(
        const pybind11::array_t<float> &input,
        const std::string &pcd_file,
        float voxelLeaf,  // 降采样阈值
        int minKNearest,    // 最小最近邻邻居的个数
        int stepSize,       // 步长
        float radius     // 最近邻搜索阈值
) {
    ME.init(voxelLeaf, minKNearest, stepSize, radius);

    if (input.size() != 0) {
        auto ref_input = input.unchecked<2>();
        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>(ref_input.shape(0), 1));
#pragma omp for schedule(dynamic)
        for (int i = 0; i < ref_input.shape(0); ++i) {
            cloud->points[i].x = ref_input(i, 0);
            cloud->points[i].y = ref_input(i, 1);
            cloud->points[i].z = ref_input(i, 2);
            cloud->points[i].intensity = ref_input(i, 3);
        }
        ME.setInputCloud(cloud);
    } else if (!pcd_file.empty()) {
        ME.setInputCloud(pcd_file);
    } else {
        std::cout << "Please set input PointCloud.\n";
    }
    ME.calculate();

}

double mme() {
    return ME.mme();
}

double mpv() {
    return ME.mpv();
}

PYBIND11_MODULE(map_evaluation, m) {
    m.doc() = "Map metrics evaluation";

    m.def("init", &init, "map evaluation",
          pybind11::arg("input")=pybind11::array_t<float>{}, pybind11::arg("pcd_file") = "", pybind11::arg("voxelLeaf")= 0.3,
          pybind11::arg("minKNearest")=5, pybind11::arg("stepSize")=1, pybind11::arg("radius")=1.0
    );

    m.def("mme", &mme, "meanMapEntropy");
    m.def("mpv", &mpv, "meanPlaneVariance");

}