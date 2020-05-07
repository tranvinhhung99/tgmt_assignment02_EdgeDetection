#include "test_utils.h"

#include <assert.h>
#include <iostream>
#include <opencv2/core.hpp>

  
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
  bool eq; 
  int mat1_depth = mat1.depth();
  int mat2_depth = mat2.depth();

  // Convert depth before compare
  if ((mat1_depth % 2 == mat2_depth % 2) && (mat1_depth != mat2_depth)){
    int max_depth = mat1_depth > mat2_depth ? mat1_depth : mat2_depth;
    cv::Mat _mat1 = mat1.getMat();
    _mat1.convertTo(_mat1, max_depth);
    cv::Mat _mat2 = mat2.getMat();
    _mat2.convertTo(_mat2, max_depth);
    eq = cv::countNonZero(_mat1 != _mat2) == 0;
  }
  else
    eq = cv::countNonZero(mat1.getMat() != mat2.getMat()) == 0;
  if(!eq){
    std::cout << "Not equal" << std::endl;
    std::cout << mat1.getMat() << std::endl;
    std::cout << mat2.getMat() << std::endl;
  }

  assert(eq);
}

