#include "utils.h"

//--------------------------------------------------------------------
// Template helper function section
//--------------------------------------------------------------------
template <typename T>
inline T getValueFromMat_(cv::InputArray input, int x, int y, int c){
  if (x < 0 || x >= input.cols())
    return 0;
  if (y < 0 || y >= input.rows())
    return 0;

  return *(input.getMat().ptr<T>(y, x, c));
}


template <typename T>
inline long long applyMask_INT(cv::InputArray src, cv::InputArray kernel, cv::Point anchor, int x, int y, int c){
  long long value = 0;
  for (int x_k = 0; x_k < kernel.cols(); x_k++)
    for (int y_k = 0; y_k < kernel.rows(); y_k++)
        value += getValueFromMat_<T>(kernel, x_k, y_k, 0)
          * getValueFromMat_<T>(src, x + x_k - anchor.x, y + y_k - anchor.y, c);
  return value;
}

template <typename T>
inline double applyMask_FLOAT(cv::InputArray src, cv::InputArray kernel, cv::Point anchor, int x, int y, int c){
  double value = 0;
  for (int x_k = 0; x_k < kernel.cols(); x_k++)
    for (int y_k = 0; y_k < kernel.rows(); y_k++)
        value += getValueFromMat_<T>(kernel, x_k, y_k, 0)
          * getValueFromMat_<T>(src, x + x_k - anchor.x, y + y_k - anchor.y, c);
  return value;
}

//--------------------------------------------------
// Main function
//-------------------------------------------------
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
        for(int c = 0; c < src.channels(); c++){
          long value; 

          value = applyMask_INT<uchar>(src, kernel, anchor, x, y, c);

          // Casting for no overflow
          *dstMat.ptr<uchar>(y, x, c) = cv::saturate_cast<uchar>(value);
        } 
}

