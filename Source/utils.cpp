#include "utils.h"
#include <iostream>

//--------------------------------------------------------------------
// Template helper function section
//--------------------------------------------------------------------
template <typename T>
inline T getValueFromMat_(cv::InputArray input, int x, int y, int c){
  if (x < 0 || x >= input.cols())
    return 0;
  if (y < 0 || y >= input.rows())
    return 0;

  int n_channel = input.channels();
  return input.getMat().at<T>(y, x*n_channel +  c);
}


template <typename T, typename K>
inline long long applyMask_INT(cv::InputArray src, cv::InputArray kernel, cv::Point anchor, int x, int y, int c){
  long long value = 0;
  for (int x_k = 0; x_k < kernel.cols(); x_k++)
    for (int y_k = 0; y_k < kernel.rows(); y_k++)
        value += (long long) getValueFromMat_<T>(kernel, x_k, y_k, 0)
          * (long long) getValueFromMat_<K>(src, x + x_k - anchor.x, y + y_k - anchor.y, c);
  return value;
}

template <typename T, typename K>
inline double applyMask_FLOAT(cv::InputArray src, cv::InputArray kernel, cv::Point anchor, int x, int y, int c){
  double value = 0;
  for (int x_k = 0; x_k < kernel.cols(); x_k++)
    for (int y_k = 0; y_k < kernel.rows(); y_k++)
        value += getValueFromMat_<T>(kernel, x_k, y_k, 0)
          * getValueFromMat_<K>(src, x + x_k - anchor.x, y + y_k - anchor.y, c);
  return value;
}

typedef long long (*pApplyMaskInt) (cv::InputArray src, cv::InputArray kernel, cv::Point anchor, int x, int y, int c);

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
  int num_channels = src.channels();
  dst.createSameSize(src, ddepth + 8 * (num_channels - 1));

  cv::Mat dstMat = dst.getMat();
  dstMat = 0;
  
  // Get variable needed for typing
  int src_depth = src.depth();
  int kernel_depth = kernel.depth();

  // for each pixel
  for (int y = 0; y < src.rows(); y++)
    for (int x = 0; x < src.cols(); x++)
        for(int c = 0; c < src.channels(); c++){
          long value; 
          switch(kernel_depth){
            case 0:
              switch(src_depth){
                case 0:
                  value = applyMask_INT<uchar, uchar>(src, kernel, anchor, x, y, c);
                  break;
                case 1:
                  value = applyMask_INT<uchar, char>(src, kernel, anchor, x, y, c);
                  break;
                case 2:
                  value = applyMask_INT<uchar, unsigned short>(src, kernel, anchor, x, y, c);
                  break;
                case 3:
                  value = applyMask_INT<uchar, short>(src, kernel, anchor, x, y, c);
                  break;
                }
                break;

            case 1:
              switch(src_depth){
                case 0:
                  value = applyMask_INT<char, uchar>(src, kernel, anchor, x, y, c);
                  break;
                case 1:
                  value = applyMask_INT<char, char>(src, kernel, anchor, x, y, c);
                  break;
                case 2:
                  value = applyMask_INT<char, unsigned short>(src, kernel, anchor, x, y, c);
                  break;
                case 3:
                  value = applyMask_INT<char, short>(src, kernel, anchor, x, y, c);
                  break;
                }
                break;

            case 2:
              switch(src_depth){
                case 0:
                  value = applyMask_INT<unsigned short, uchar>(src, kernel, anchor, x, y, c);
                  break;
                case 1:
                  value = applyMask_INT<unsigned short, char>(src, kernel, anchor, x, y, c);
                  break;
                case 2:
                  value = applyMask_INT<unsigned short, unsigned short>(src, kernel, anchor, x, y, c);
                  break;
                case 3:
                  value = applyMask_INT<unsigned short, short>(src, kernel, anchor, x, y, c);
                  break;
                }

            case 3:
              switch(src_depth){
                case 0:
                  value = applyMask_INT<short, uchar>(src, kernel, anchor, x, y, c);
                  break;
                case 1:
                  value = applyMask_INT<short, char>(src, kernel, anchor, x, y, c);
                  break;
                case 2:
                  value = applyMask_INT<short, unsigned short>(src, kernel, anchor, x, y, c);
                  break;
                case 3:
                  value = applyMask_INT<short, short>(src, kernel, anchor, x, y, c);
                  break;
                }
                break;
          }
          // Casting for no overflow
          switch(ddepth){
            case 0:
              dstMat.at<uchar>(y, x*num_channels + c) = cv::saturate_cast<uchar>(value);
              break;
            case 1:
              dstMat.at<char>(y, x*num_channels + c) = cv::saturate_cast<char>(value);
              break;
            case 2:
              dstMat.at<unsigned short>(y, x*num_channels + c) = cv::saturate_cast<unsigned short>(value);
              break;
            case 3:
              dstMat.at<short>(y, x*num_channels + c) = cv::saturate_cast<short>(value);
              break;
          }
        } 
}

