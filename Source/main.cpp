#include "utils.h"

#include <iostream>
#include <opencv2/core.hpp>


void fill_array(int *a, int length, int fill_value){
  for (int i = 0; i < length; i++)
    a[i] = fill_value;
}
  

int main(){
  int input_data[9];
  fill_array(input_data, 9, 0);

  int kernel_data[9];
  fill_array(kernel_data, 9, 1);

  cv::Mat src(3, 3, CV_8UC1, input_data);
  cv::Mat kernel(3, 3, CV_8UC1, kernel_data);

  cv::Mat out;
  utils::applyFilter(src, out, -1, kernel);
  std::cout << out << std::endl;
}
