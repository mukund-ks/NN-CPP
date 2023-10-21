#pragma once
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

typedef std::vector<std::pair<Eigen::RowVectorXf, std::vector<int>>> Set;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowMatrix;
typedef cv::Mat cvImg;

class Dataset {
   private:
    Set data;
    Set trainingSet;
    Set validationSet;
    Set testingSet;

    std::string dataDir;
    int numClasses;

    void loadData();
    void createSplits();
    void shuffleDataset();
    void getEigenImage(const cvImg& image, RowMatrix& result);
    void oneHotEncode(const int& classId, std::vector<int>& labelVec);

   public:
    Dataset(std::string dataDir, int numClasses);
    void prepareDataset();
    const Set& getTrainingSet() const;
    const Set& getValidationSet() const;
    const Set& getTestingSet() const;
};