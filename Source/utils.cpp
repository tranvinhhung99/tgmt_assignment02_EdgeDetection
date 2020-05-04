#include "utils.h"


int getValueFromInput(cv::InputArray input, int x, int y, int c){
  if (x < 0 || x >= input.cols())
    return 0;
  if (y < 0 || y >= input.rows())
    return 0;

  return input.getMat().at<cv::Vec3b>(y, x)[c];
}


void utils::applyFilter(cv::InputArray src,
    cv::OutputArray dst,
    int ddepth,
    cv::InputArray kernel,
    cv::Point anchor
){
  // Getting true value of -1 value in default:
  if (ddepth == -1)
    ddepth = src.depth();

  if (anchor.x == -1)
    anchor.x = kernel.rows() / 2;

  if (anchor.y == -1)
    anchor.y = kernel.cols() / 2;

  // Init destination 
  dst.createSameSize(src, ddepth);

  cv::Mat dstMat = dst.getMat();
  dstMat = 0;
  // for each pixel
  for (int y = 0; y < src.rows(); y++)
    for (int x = 0; x < src.cols(); x++)
      for (int c = 0; c < src.channels(); c++)
        // for each value in kernel
        for (int x_k = 0; x_k < kernel.cols(); x_k++)
          for (int y_k = 0; y_k < kernel.rows(); y_k++)
            dstMat.at<cv::Vec3b>(y, x)[c] += getValueFromInput(kernel, x_k, y_k, 0)
              * getValueFromInput(src, x + x_k - anchor.x, y + y_k - anchor.y, c);
}
