//
// Created by hyx on 2021/11/11.
//

#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <string>
#include <omp.h>

#include <ctime>
#include <cstdlib>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/common/geometry.h>
#include <pcl/common/centroid.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/console/parse.h>

typedef pcl::PointXYZI PointType;


double Mean_Map_Entropy(Eigen::Matrix3d &covarianceMatrix) {
    // 求协方差矩阵的熵
    double det = (2 * M_PI * M_E * covarianceMatrix).determinant();
    if (det >= 0) {
        return 0.5 * std::log(det);
    }
    return std::numeric_limits<double>::infinity();
}

double Mean_Plane_Variance(Eigen::Matrix3d &covarianceMatrix) {
    // 求解特征值
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(covarianceMatrix);
    if (eigenSolver.info() == Eigen::Success) {
        // 最小的特征值对应方差最小的方向，即点集构成平面的法向量
        // note Eigen library sort eigenvalues in increasing order
        if (3 * eigenSolver.eigenvalues()[0] < eigenSolver.eigenvalues()[1])
            return eigenSolver.eigenvalues()[0];
    }
    return std::numeric_limits<double>::infinity();
}


int main(int argc, char **argv) {

    std::string pcd_file = std::string(argv[1]);

    pcl::PointCloud<PointType>::Ptr inputCloud(new pcl::PointCloud<PointType>());
    //    pcl::PointCloud<PointType>::Ptr inputCloudDS(new pcl::PointCloud<PointType>());
    // 加载点云pcd
    if (pcl::io::loadPCDFile<PointType>(pcd_file, *inputCloud) == -1) {
        PCL_ERROR("Couldn't open this pc file! Please check out your file path.\n");
        return -1;
    }

    double entropySum = 0.f;
    double planeVarianceSum = 0.f;

    // 设置参数
    float voxelLeaf = 0.3;  // 降采样阈值
    int minKNearest = 5;    // 最小最近邻邻居的个数
    int stepSize = 5;       // 步长
    float radius = 1.0;     // 最近邻搜索阈值
    int searchNeighbors = 10;

    // 参数可从终端输入
    pcl::console::parse_argument(argc, argv, "--step", stepSize);
    pcl::console::parse_argument(argc, argv, "--radius", radius);
    pcl::console::parse_argument(argc, argv, "--downSample", voxelLeaf);

    if (pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "--help")) {
        //        showHelp(argv[0]);
        return 0;
    }
    if (voxelLeaf != 0) {   // 降采样
        pcl::VoxelGrid<PointType> filter;
        filter.setLeafSize(voxelLeaf, voxelLeaf, voxelLeaf);
        filter.setInputCloud(inputCloud);
        filter.filter(*inputCloud);
    }

    pcl::KdTreeFLANN<PointType> kdtree;
    kdtree.setInputCloud(inputCloud);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
#pragma omp parallel reduction (+:entropySum, planeVarianceSum)
    {
//        planeEstimate()
#pragma omp for schedule(dynamic)
        for (std::size_t i = 0; i < inputCloud->points.size(); i += stepSize) {
            // print status
            if (i % (inputCloud->points.size() / 20) == 0) {
                int percent = i * 100 / inputCloud->points.size();
                std::cout << percent << " %" << std::endl;
            }

            // 寻找最近邻点
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            int numberOfNeighbors = kdtree.radiusSearch(inputCloud->points[i], radius, pointIdxRadiusSearch,
                                                        pointRadiusSquaredDistance);

            // 寻找是否具有足够数量的邻居
            double mme = .0f, mpv = .0f;
            if (numberOfNeighbors > minKNearest) {
                // 保存邻居点
                pcl::PointCloud<PointType>::Ptr neighbors(new pcl::PointCloud<PointType>());
                for (std::size_t t = 0; t < pointIdxRadiusSearch.size(); t++) {
                    neighbors->points.push_back(inputCloud->points[pointIdxRadiusSearch[t]]);
                }

                Eigen::Vector4d centroid;
                Eigen::Matrix3d covarianceMatrix = Eigen::Matrix3d::Identity();

                // 求点集质心和协方差矩阵
                pcl::compute3DCentroid(*neighbors, centroid);
                pcl::computeCovarianceMatrixNormalized(*neighbors, centroid, covarianceMatrix);

                // 求解mme、mpv
                mme = Mean_Map_Entropy(covarianceMatrix);
                mpv = Mean_Plane_Variance(covarianceMatrix);

            } else {
                mme = std::numeric_limits<double>::infinity();
                mpv = std::numeric_limits<double>::infinity();
            }

            if (std::isfinite(mme)) entropySum += mme;

            if (std::isfinite(mpv)) planeVarianceSum += mpv;
        }
    }
    // 计算均值
    double meanMapEntropy = entropySum / (static_cast<double>(inputCloud->points.size() / stepSize));
    double meanPlaneVariance = planeVarianceSum / (static_cast<double>(inputCloud->points.size() / stepSize));
    // 计算时间
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "------------------- " << std::endl;
    std::cout << "Mean Map Entropy is " << meanMapEntropy << std::endl;
    std::cout << "Mean Plane Variance is " << meanPlaneVariance << std::endl;

    std::cout << "Used " << elapsed_seconds.count() << " milliseconds to compute values for "
    << inputCloud->points.size() << " points." << std::endl;


    return 0;
}