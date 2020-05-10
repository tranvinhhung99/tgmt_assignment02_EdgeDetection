#include "utils.h"
#include "canny.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <string.h>

const int BORDER_CONSTANT = 0;

// Show help prompt
void printHelp(){
  std::cout << "Edge Detection program" << std::endl
    << "Usage: <program_name> <image_path> <command_id> <extra_params>" << std::endl
    << "  " << "program_name: Name of this program" << std::endl
    << "  " << "image_path: Path to image need to detect edge" << std::endl
    << "  " << "command_id: SOBEL, PREWITT, LAPLACE, CANNY" << std::endl
    << "  " << "SOBEL and PREWITT extra params: 0 or 1 (default: 0).1: Show grad_x and grad_y" << std::endl
    << "  " << "LAPLACE extra params: currently none" << std::endl
    << "  " << "CANNY extra params: " << std::endl
    << "  " << "      - low_thres: Low threshold (default:100)" << std::endl
    << "  " << "      - high_thres: High threshold (default:200)" << std::endl
    << std::endl;

}


int main(int argc, const char** argv){
  if(argc < 3 || argc > 5){
    printHelp();
    return 0;
  }

  //Try read image
  cv::Mat img;
  img = cv::imread(argv[1]);
  if(img.empty()){
    std::cout << "[ERROR]: Cannot open" << argv[1] << std::endl;
    return 1;
  }

  if(img.channels() > 1)
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

  if(strcmp(argv[2], "SOBEL") == 0){
    cv::Mat edge, grad_x, grad_y;
    utils::detectBySobel(img, edge, grad_x, grad_y);

    cv::imshow("Edge by Sobel", edge);

    bool flag = argc == 3;

    if(!flag)
      flag = argv[3][0] == '1';

    if(flag){
      cv::imshow("Grad X", grad_x);
      cv::imshow("Grad y", grad_y);
    }
  }
  else if(strcmp(argv[2], "PREWITT") == 0){
    cv::Mat edge, grad_x, grad_y;
    utils::detectBySobel(img, edge, grad_x, grad_y);

    cv::imshow("Edge by Prewitt", edge);

    bool flag = argc == 3;

    if(!flag)
      flag = argv[3][0] == '1';

    if(flag){
      cv::imshow("Grad X", grad_x);
      cv::imshow("Grad y", grad_y);
    }
  }
  else if(strcmp(argv[2], "LAPLACE") == 0){
    cv::Mat edge;
    utils::detectByLaplace(img, edge);

    cv::imshow("Edge by Laplace", edge);

  }
  else if(strcmp(argv[2], "CANNY") == 0){
    cv::Mat edge;
    int low_thres = 100, high_thres = 200;
    if(argc >= 4)
      low_thres = atoi(argv[3]);
    if(argc >= 5)
      high_thres = atoi(argv[4]);
    utils::detectByCanny(img, edge, low_thres, high_thres);
    cv::imshow("Edge by Canny", edge);
  }



}
