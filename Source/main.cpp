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
    << "  " << "      - low_thres: Low threshold (default:40)" << std::endl
    << "  " << "      - high_thres: High threshold (default:80)" << std::endl
    << "  " << "      - kernel_size_gauss: GaussianFilter's kernel size (default:3)" << std::endl
    << "  " << "      - kernel_size_nms: NonMaxSupression's kernel size (default:3)" << std::endl
    << std::endl;

}


int main(int argc, const char** argv){
  if(argc < 3 || argc > 7){
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

  cv::imshow("Original Image", img);

  if(img.channels() > 1)
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

  if(strcmp(argv[2], "SOBEL") == 0){
    cv::Mat edge, grad_x, grad_y;
    utils::detectBySobel(img, edge, grad_x, grad_y);

    edge.convertTo(edge, CV_8U);
    cv::imshow("Edge by Sobel", edge);

    bool flag = argc == 3;

    if(!flag)
      flag = argv[3][0] == '0';

    if(!flag){
      grad_x.convertTo(grad_x, CV_8U);
      cv::imshow("Grad X", grad_x);
      grad_y.convertTo(grad_y, CV_8U);
      cv::imshow("Grad y", grad_y);
    }
  }
  else if(strcmp(argv[2], "PREWITT") == 0){
    cv::Mat edge, grad_x, grad_y;
    utils::detectBySobel(img, edge, grad_x, grad_y);

    edge.convertTo(edge, CV_8U);
    cv::imshow("Edge by Prewitt", edge);

    bool flag = argc == 3;

    if(!flag)
      flag = argv[3][0] == '0';

    if(!flag){
      grad_x.convertTo(grad_x, CV_8U);
      cv::imshow("Grad X", grad_x);
      grad_y.convertTo(grad_y, CV_8U);
      cv::imshow("Grad y", grad_y);
    }
  }
  else if(strcmp(argv[2], "LAPLACE") == 0){
    cv::Mat edge;
    utils::detectByLaplace(img, edge);

    edge.convertTo(edge, CV_8U);
    cv::imshow("Edge by Laplace", edge);

  }
  else if(strcmp(argv[2], "CANNY") == 0){
    cv::Mat edge;
    int low_thres = 40, high_thres = 80, kernel_size_gauss = 3, kernel_size_nms = 5;

    switch (argc)
    {
    case 7:
        kernel_size_nms = atoi(argv[6]);
    case 6:
        kernel_size_gauss = atoi(argv[5]);
    case 5:
        high_thres = atoi(argv[4]);
    case 4:
        low_thres = atoi(argv[3]);
    case 3:
        utils::detectByCanny(img, edge, low_thres, high_thres, kernel_size_gauss, kernel_size_nms);
        break;
    default:
        std::cout << "Not support CANNY with " << argc - 3 << " parameters!" << std::endl;
    }

    edge.convertTo(edge, CV_8U);
    cv::imshow("Edge by Canny", edge);
  }
  else{
    std::cout << "Not support " << argv[2] << std::endl;
    printHelp();
    return 1;
  }
  //std::cout << "Press any key to end" << std::endl;

  cv::waitKey(0);
  return 0;
}
