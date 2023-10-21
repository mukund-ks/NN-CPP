#include "data_loader.h"

#include <filesystem>
#include <opencv2/core/eigen.hpp>
#include <random>
namespace fs = std::filesystem;

Dataset::Dataset(std::string dataDir, int numClasses) {
    this->dataDir = dataDir;
    this->numClasses = numClasses;
}

void Dataset::getEigenImage(const cvImg& image, RowMatrix& result) {
    Matrix eigenImg;
    cv::cv2eigen(image, eigenImg);
    result = eigenImg.reshaped<Eigen::RowMajor>().transpose();
}

void Dataset::oneHotEncode(const int& classId, std::vector<int>& labelVec) {
    labelVec.resize(numClasses, 0);
    labelVec[classId] = 1;
    return;
}

void Dataset::loadData() {
    if (!fs::exists(dataDir)) {
        throw std::runtime_error("Provided Directory for data: " + dataDir + ", does not exist.");
    }

    for (const auto& classDirEntry : fs::directory_iterator(dataDir)) {
        int classId = std::stoi(classDirEntry.path().filename().string());
        if (!fs::is_directory(classDirEntry)) continue;
        for (auto& entry : fs::directory_iterator(classDirEntry)) {
            if (entry.path().extension() == ".png") {
                cvImg image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
                if (!image.empty()) {
                    std::vector<int> labelVec(numClasses);
                    RowMatrix imageVec;
                    oneHotEncode(classId, labelVec);
                    getEigenImage(image, imageVec);
                    data.push_back(std::make_pair(imageVec, labelVec));
                }
            }
        }
    }
    return;
}

void Dataset::shuffleDataset() {
    std::random_device rd;
    std::default_random_engine ran_eg(rd());
    std::shuffle(data.begin(), data.end(), ran_eg);
    return;
}

void Dataset::createSplits() {
    int totalSize = data.size();
    int trainSize = static_cast<int>(totalSize * 0.6);
    int validationSize = static_cast<int>(totalSize * 0.2);

    trainingSet.assign(data.begin(), data.begin() + trainSize);
    validationSet.assign(data.begin() + trainSize, data.begin() + trainSize + validationSize);
    testingSet.assign(data.begin() + trainSize + validationSize, data.end());
    return;
}

void Dataset::prepareDataset() {
    loadData();
    shuffleDataset();
    createSplits();
}

const Set& Dataset::getTrainingSet() const {
    return trainingSet;
}

const Set& Dataset::getValidationSet() const {
    return validationSet;
}

const Set& Dataset::getTestingSet() const {
    return testingSet;
}