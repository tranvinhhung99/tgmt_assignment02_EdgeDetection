#include "utils.h"
#include "test_utils.h"
#include "generators.h"

#include "gtest/gtest.h"
#include <assert.h>
#include <opencv2/imgproc.hpp>


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

TEST(_2DFilterTest, randomTest_uint8){
  cv::Mat src;
  test_case::generateRandomMatrix(src, 30, 30);
  cv::Mat kernel;
  test_case::generateRandomMatrix(kernel, 3, 3);

  cv::Mat my_output;
  utils::applyFilter(src, my_output, -1, kernel);

  cv::Mat cv_output;
  cv::filter2D(src, cv_output, -1, kernel);

  test_case::testSameMatrix(my_output, cv_output);


}

