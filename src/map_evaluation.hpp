//
// Created by hyx on 2021/11/18.
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

class map_evaluation {
private:
    // mme, mpv
    double meanMapEntropy = .0f, meanPlaneVariance = .0f;
    // 参数设置
    float voxelLeaf;  // 降采样阈值
    int minKNearest;    // 最小最近邻邻居的个数
    int stepSize;       // 步长
    float radius;     // 最近邻搜索阈值

    pcl::PointCloud<PointType>::Ptr inputCloud;
    pcl::KdTreeFLANN<PointType> kdtree;

public:
    map_evaluation() {}
    ~map_evaluation() {}

    void init(float _voxelLeaf = 0.3, int _minKNearest = 5, int _stepSize = 1, float _radius = 1.0){
        inputCloud.reset(new pcl::PointCloud<PointType>());

        voxelLeaf = _voxelLeaf;
        minKNearest = _minKNearest;
        stepSize = _stepSize;
        radius = _radius;

        std::cout << "-----initialization succeed!-----\n";
    }

    // 读pcd文件
    void setInputCloud(const std::string &pcd_file) {
        if (pcl::io::loadPCDFile<PointType>(pcd_file, *inputCloud) == -1) {
            PCL_ERROR("Couldn't open this pc file! Please check out your file path.\n");
            exit(-1);
        }
    }

    // 直接拷贝输入点云
    void setInputCloud(pcl::PointCloud<PointType>::ConstPtr cloudIn) {
        pcl::copyPointCloud(*cloudIn, *inputCloud);
    }

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

    void calculate() {

        if (voxelLeaf != 0) {   // 降采样
            pcl::VoxelGrid<PointType> filter;
            filter.setLeafSize(voxelLeaf, voxelLeaf, voxelLeaf);
            filter.setInputCloud(inputCloud);
            filter.filter(*inputCloud);
        }

        kdtree.setInputCloud(inputCloud);

        double entropySum = 0.f;
        double planeVarianceSum = 0.f;

        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();

#pragma omp parallel reduction (+:entropySum, planeVarianceSum)
        {
#pragma omp for schedule(dynamic)
            for (std::size_t i = 0; i < inputCloud->points.size(); i += stepSize) {
                // print status
//                if (i % (inputCloud->points.size() / 20) == 0) {
//                    int percent = i * 100 / inputCloud->points.size();
//                    std::cout << percent << " %" << std::endl;
//                }

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
        meanMapEntropy = entropySum / (static_cast<double>(inputCloud->points.size() / stepSize));
        meanPlaneVariance = planeVarianceSum / (static_cast<double>(inputCloud->points.size() / stepSize));

        // 计算时间
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

//        std::cout << "------------------- " << std::endl;
//        std::cout << "Mean Map Entropy is " << meanMapEntropy << std::endl;
//        std::cout << "Mean Plane Variance is " << meanPlaneVariance << std::endl;

        std::cout << "Used " << elapsed_seconds.count() << " seconds to compute values for "
        << inputCloud->points.size() << " points." << std::endl;
    }

    double mme(){
        return meanMapEntropy;
    }

    double mpv(){
        return meanPlaneVariance;
    }

};

