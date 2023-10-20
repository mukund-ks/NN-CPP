#pragma once
#include <opencv4/opencv2/opencv.hpp>
#include <utility>
#include <vector>

typedef std::vector<std::pair<cv::Mat, std::vector<int>>> Set;

class Dataset {
   private:
    std::vector<cv::Mat> image;
    std::vector<int> label;

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
    std::vector<int> oneHotEncode(int label);

   public:
    Dataset(const std::string& dataDir, int numClasses, int trainSize, int validationSize, int testSize);
    void prepareDataset();
    const Set& getTrainingSet() const;
    const Set& getValidationSet() const;
    const Set& getTestingSet() const;
};