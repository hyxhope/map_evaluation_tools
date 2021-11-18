//
// Created by hyx on 2021/11/11.
//

#include "map_evaluation.hpp"

int main(int argc, char **argv) {

    std::string pcd_file = std::string(argv[1]);
    pcl::PointCloud<PointType>::Ptr inputCloud(new pcl::PointCloud<PointType>());

    if (pcl::io::loadPCDFile<PointType>(pcd_file, *inputCloud) == -1) {
        PCL_ERROR("Couldn't open this pc file! Please check out your file path.\n");
        return -1;
    }

    map_evaluation ME;

    ME.setInputCloud(inputCloud);
    ME.calculate();

    std::cout << "------------------- " << std::endl;
    std::cout << "Mean Map Entropy is " << ME.mme() << std::endl;
    std::cout << "Mean Plane Variance is " << ME.mpv() << std::endl;

    return 0;
}