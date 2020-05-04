#include "test_utils.h"

#include <assert.h>
#include <iostream>

  
// Checking if two matrix is the same size
void test_case::testSameSize(cv::InputArray mat1, cv::InputArray mat2){
  assert(mat1.cols() == mat2.cols());
  assert(mat1.rows() == mat2.rows());
  assert(mat1.channels() == mat2.channels());
}

void test_case::testIsFilledWithConst(cv::InputArray input, const uchar value){
  int num_lines = input.rows();
  // Get number element per line
  int num_elements = input.cols() * input.channels(); 

  for (int i = 0; i < num_lines; i++){
    uchar* data = input.getMat().ptr<uchar>(i);
    for (int j = 0; j < num_elements; j++)
      assert(data[j] == value);
  }
}

void test_case::testSameMatrix(cv::InputArray mat1, cv::InputArray mat2){
  bool eq = cv::countNonZero(mat1 != mat2) == 0;
  assert(eq);
}

