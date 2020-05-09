#include "utils.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

const int BORDER_CONSTANT = 0;


void fill_array(int *a, int length, int fill_value){
  for (int i = 0; i < length; i++)
    a[i] = fill_value;
}
  

int main(int argc, const char** argv){
  //int input_data[9];
  //fill_array(input_data, 9, 0);

  //int kernel_data[9];
  //fill_array(kernel_data, 9, 1);

  //cv::Mat src(3, 3, CV_8UC1, input_data);
  //cv::Mat kernel(3, 3, CV_8UC1, kernel_data);

  //cv::Mat out;
  //utils::applyFilter(src, out, -1, kernel);
  //std::cout << out << std::endl;
  
  cv::Mat img;
  img = cv::imread(argv[1]);
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  utils::applyGaussianFilter(img, img, 3);

  cv::Mat edge, grad_x, grad_y;
  utils::detectBySobel(img, edge, grad_x, grad_y);

  cv::imwrite("sample_output/grad_x.jpg", grad_x);
  cv::imwrite("sample_output/grad_y.jpg", grad_y);
  cv::imwrite("sample_output/edge.jpg", edge);

  cv::Mat cv_grad_x, cv_grad_y;
  cv::Sobel(img, cv_grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_CONSTANT);
  cv::Sobel(img, cv_grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_CONSTANT);
  cv::imwrite("sample_output/cv_grad_x.jpg", cv_grad_x);
  cv::imwrite("sample_output/cv_grad_y.jpg", cv_grad_y);

  cv::Mat abs_grad_x, abs_grad_y;
  cv::convertScaleAbs( grad_x, abs_grad_x );
  cv::convertScaleAbs( grad_y, abs_grad_y );
  cv::Mat grad;
  cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

  cv::imwrite("sample_output/edge_abs.jpg", grad);


}
