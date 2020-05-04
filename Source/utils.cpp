#include "utils.h"


//-----------------------------------------------------
// Warper for greyscale image section
//----------------------------------------------------
inline int getValueFromMat(cv::InputArray input, int x, int y){
  if (x < 0 || x >= input.cols())
    return 0;
  if (y < 0 || y >= input.rows())
    return 0;

  return input.getMat().at<uchar>(y, x);
}


inline long applyMask(cv::InputArray src, cv::InputArray kernel, cv::Point anchor, int x, int y){
  long value = 0;
  for (int x_k = 0; x_k < kernel.cols(); x_k++)
    for (int y_k = 0; y_k < kernel.rows(); y_k++)
        value += getValueFromMat(kernel, x_k, y_k)
          * getValueFromMat(src, x + x_k - anchor.x, y + y_k - anchor.y);
  return value;
}

//-----------------------------------------------------
// Warper for color image section
//----------------------------------------------------
inline int getValueFromMat(cv::InputArray input, int x, int y, int c){
  if (x < 0 || x >= input.cols())
    return 0;
  if (y < 0 || y >= input.rows())
    return 0;

  return input.getMat().at<cv::Vec3b>(y, x)[c];
}

inline long applyMask(cv::InputArray src, cv::InputArray kernel, cv::Point anchor, int x, int y, int c){
  long value = 0;
  for (int x_k = 0; x_k < kernel.cols(); x_k++)
    for (int y_k = 0; y_k < kernel.rows(); y_k++)
        value += getValueFromMat(kernel, x_k, y_k, c)
          * getValueFromMat(src, x + x_k - anchor.x, y + y_k - anchor.y, c);
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
      /* Greyscale different from colors
       * image due to value type.
       * 
       * Greyscale use uchar.
       * Color use Vec3b.
       */
      if (src.channels() == 1){ // Greyscale
        long value = applyMask(src, kernel, anchor, x, y);
        
        // Casting for no overflow
        dstMat.at<uchar>(y, x) = cv::saturate_cast<uchar>(value);
      }
      else if (src.channels() == 3){
        for(int c = 0; c < 3; c++){
          long value = applyMask(src, kernel, anchor, x, y, c);

          // Casting for no overflow
          dstMat.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(value);
        } 
      }
}

