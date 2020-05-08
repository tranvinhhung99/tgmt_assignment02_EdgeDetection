#include "utils.h"
#include "test_utils.h"
#include "generators.h"

#include "gtest/gtest.h"
#include <assert.h> 
#include <opencv2/imgproc.hpp>
#include <iostream>


const int BORDER_CONSTANT = 0;
void fillArray(int *a, int length, int fill_value){
  for (int i = 0; i < length; i++)
    a[i] = fill_value;
}

  
void testEmptyArr(){
  int input_data[9];
  fillArray(input_data, 9, 0);

  int kernel_data[9];
  fillArray(kernel_data, 9, 1);

  cv::Mat src(3, 3, CV_8UC1, input_data);
  cv::Mat kernel(3, 3, CV_8UC1, kernel_data);

  cv::Mat out;
  utils::applyFilter(src, out, -1, kernel);

  test_case::testSameSize(out, src);
  test_case::testIsFilledWithConst(out, 0);
  

}

TEST(_2DFilterTest, emptyInputTest){
  testEmptyArr();
}

TEST(_2DFilterTest, knownInput_uint8){
  int src_data[9] = {3, 0, 2, 4, 0, 3, 1, 0, 1};
  cv::Mat src(3, 3, CV_8UC1, &src_data);
  int kernel_data[9] = {1, 1, 1, 0, 0, 0, 1, 0, 1};
  cv::Mat kernel(3, 3, CV_8UC1, &kernel_data);

  cv::Mat my_output;
  utils::applyFilter(src, my_output, -1, kernel);

  cv::Mat cv_output;
  cv::filter2D(src, cv_output, -1, kernel, cv::Point(-1, -1), 0, BORDER_CONSTANT);

  test_case::testSameMatrix(my_output, cv_output);

}


TEST(_2DFilterTest, randomTest_uint8){
  cv::Mat src;
  test_case::generateRandomMatrix(src, 30, 30);
  cv::Mat kernel;
  test_case::generateRandomMatrix(kernel, 3, 3);

  cv::Mat my_output;
  utils::applyFilter(src, my_output, -1, kernel);

  cv::Mat cv_output;
  cv::filter2D(src, cv_output, -1, kernel, cv::Point(-1, -1), 0, BORDER_CONSTANT);

  test_case::testSameMatrix(my_output, cv_output);

}

TEST(_2DFilterTest, knownKernel_uint8){
  cv::Mat src;
  test_case::generateRandomMatrix(src, 30, 30);
  int kernel_data[9] = {1, 1, 1, 0, 0, 0, 1, 0, 1};
  cv::Mat kernel(3, 3, CV_8UC1, &kernel_data);

  cv::Mat my_output;
  utils::applyFilter(src, my_output, -1, kernel);

  cv::Mat cv_output;
  cv::filter2D(src, cv_output, -1, kernel, cv::Point(-1, -1), 0, BORDER_CONSTANT);

  test_case::testSameMatrix(my_output, cv_output);

}

void printArray(cv::Mat src){
  int nl = src.rows;
  int ne = src.cols * src.channels();

  for(int i = 0; i < nl; i++){
    uchar* data = src.ptr<uchar>(i);
    for(int j = 0; j < ne; j++)
      std::cerr << int(data[j]) << " ";
    std::cerr << std::endl;
  }
      
}

TEST(_2DFilterTest, randomTest_uint8_c3){
  cv::Mat src;
  test_case::generateRandomMatrix(src, 5, 5, CV_8UC3);
  cv::Mat kernel;
  test_case::generateRandomMatrix(kernel, 3, 3);
  
  cv::Mat my_output;
  utils::applyFilter(src, my_output, -1, kernel);

  cv::Mat cv_output;
  cv::filter2D(src, cv_output, -1, kernel, cv::Point(-1, -1), 0, BORDER_CONSTANT);


  test_case::testSameMatrix(my_output, cv_output);
}

TEST(_2DFilterTest, randomTest_uint8_int8){
  cv::Mat src;
  test_case::generateRandomMatrix(src, 5, 5, CV_16SC1);
  cv::Mat kernel;
  test_case::generateRandomMatrix(kernel, 3, 3, CV_16SC1);
  
  cv::Mat my_output;
  utils::applyFilter(src, my_output, 3, kernel);

  cv::Mat cv_output;
  cv::filter2D(src, cv_output, -1, kernel, cv::Point(-1, -1), 0, BORDER_CONSTANT);


  test_case::testSameMatrix(my_output, cv_output);
}

TEST(Sobel, createFilterX){
  cv::Mat kernel;
  utils::createSobelFilter(kernel, 0);

  char data[9] = {-1, 0, 1,
                   -2, 0, 2,
                   -1, 0, 1}; 
  cv::Mat ans(3, 3, CV_8S, &data);
  test_case::testSameMatrix(kernel, ans);
}

TEST(Sobel, createFilterY){
  cv::Mat kernel;
  utils::createSobelFilter(kernel, 90);

  char data[9] = {-1, -2, -1,
                    0,  0,  0,
                    1,  2,  1}; 
  cv::Mat ans(3, 3, CV_8S, &data);
  test_case::testSameMatrix(kernel, ans);
}

TEST(Sobel, detectEdge){
  cv::Mat random_mat;
  test_case::generateRandomMatrix(random_mat, 10, 10, CV_8U);

  cv::Mat my_grad_x, my_grad_y, my_edge;
  utils::detectBySobel(random_mat, my_edge, my_grad_x, my_grad_y);

  cv::Mat cv_grad_x, cv_grad_y;
  cv::Sobel(random_mat, cv_grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_CONSTANT);
  cv::Sobel(random_mat, cv_grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_CONSTANT);

  test_case::testSameMatrix(cv_grad_x, my_grad_x);
  test_case::testSameMatrix(cv_grad_y, my_grad_y);

}

