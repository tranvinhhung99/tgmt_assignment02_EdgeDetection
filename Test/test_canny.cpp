#include "utils.h"
#include "canny.h"
#include "test_utils.h"
#include "generators.h"

#include "gtest/gtest.h"
#include <assert.h> 
#include <opencv2/imgproc.hpp>
#include <iostream>

TEST(Canny, testRandom){
  cv::Mat src;
  test_case::generateRandomMatrix(src, 5, 5, CV_8UC1);


  cv::Mat my_output;
  utils::detectByCanny(src, my_output, 20, 40);

  cv::Mat cv_output;

  cv::Mat grad_x, grad_y, edge;
  cv::Mat src2;
  utils::applyGaussianFilter(src, src2, 3);
  utils::detectBySobel(src2, edge, grad_x, grad_y);
  //cv::Canny(src, cv_output, 20, 40, 3, true);
  cv::Canny(grad_x, grad_y, cv_output, 20.d, 40.d, true);

  test_case::testSameMatrix(cv_output, my_output);
  


}
