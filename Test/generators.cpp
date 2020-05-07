#include "generators.h"

#include <opencv2/core/traits.hpp>

void test_case::generateRandomMatrix(cv::OutputArray dest, int num_row, int num_col, int type){
  dest.create(num_row, num_col, type);
  int ddepth = dest.depth();

  // Random based on depth
  switch(ddepth){
    case (cv::DataDepth<float>::value):
    case (cv::DataDepth<double>::value):
      //Random uniform, value from 0.0 to 1.0
      cv::randu(dest.getMatRef(), cv::Scalar(0.0f), cv::Scalar(1.0f));
      break;
    case (cv::DataDepth<unsigned char>::value):
    case (cv::DataDepth<int>::value):
    default:
     //Random uniform, value from 0 to 255
     cv::randu(dest.getMatRef(), cv::Scalar(0), cv::Scalar(256)); 
     break;
  }
}

void test_case::generateRandomMatrix(cv::OutputArray dest, cv::Size size, int type){
  generateRandomMatrix(dest, size.height, size.width, type);
}

