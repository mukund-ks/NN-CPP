#pragma once
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

typedef std::vector<std::pair<Eigen::MatrixXf, std::vector<int>>> Set;
typedef Eigen::MatrixXf Matrix;
typedef cv::Mat cvImg;

class Dataset {
   private:
    cvImg image;
    int label;

    Set trainingSet;
    Set validationSet;
    Set testingSet;

    int numClasses;
    int trainSize;
    int validationSize;
    int testSize;

    void loadData(const std::string& dataDir);
    void createSplits();
    void shuffleDataset();
    Matrix getEigenImage(cvImg image);
    std::vector<int> oneHotEncode(int label);

   public:
    Dataset(const std::string& dataDir, int numClasses, int trainSize, int validationSize, int testSize);
    void prepareDataset();
    const Set& getTrainingSet() const;
    const Set& getValidationSet() const;
    const Set& getTestingSet() const;
};