#include "utils.h"

#include <iostream>
#include <array>

#include <math.h>
#include <opencv2/core.hpp>

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


/* Get L2 value from short Grad_X and short Grad_Y to Short dst */
void getL2(cv::InputArray grad_x, cv::InputArray grad_y, cv::OutputArray dst){
  dst.createSameSize(grad_x, CV_16S);
  for(int y = 0; y < grad_x.rows(); y++)
    for(int x = 0; x < grad_x.cols(); x++){
      long grad_x_value = grad_x.getMat().at<short>(y, x);
      long grad_y_value = grad_y.getMat().at<short>(y, x);

      double square_l2 = grad_x_value * grad_x_value + grad_y_value * grad_y_value;
      
      dst.getMat().at<short>(y, x) = cv::saturate_cast<short>(sqrt(square_l2));
    }
}

/* Big switch case function to handle multiple depth */
void applyMaskIntWarper(cv::Mat dstMat, cv::InputArray src, 
    cv::InputArray kernel, int ddepth, 
    int y, int x, int c, 
    cv::Point anchor){
  int kernel_depth = kernel.depth();
  int src_depth = src.depth();
  long value; 
  int num_channels = dstMat.channels();
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
/* Apply float warper (Big switch case function) */
void applyMaskFloatWarper(cv::Mat dstMat, cv::InputArray src, 
    cv::InputArray kernel, int ddepth, 
    int y, int x, int c, 
    cv::Point anchor){
  int kernel_depth = kernel.depth();
  int src_depth = src.depth();
  double value; 
  int num_channels = dstMat.channels();
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
    case CV_32F:
      switch(src_depth){
        case 0:
          value = applyMask_FLOAT<float, uchar>(src, kernel, anchor, x, y, c);
          break;
        case 1:
          value = applyMask_FLOAT<float, char>(src, kernel, anchor, x, y, c);
          break;
        case 2:
          value = applyMask_FLOAT<float, unsigned short>(src, kernel, anchor, x, y, c);
          break;
        case 3:
          value = applyMask_FLOAT<float, short>(src, kernel, anchor, x, y, c);
          break;
        case CV_32F:
          value = applyMask_FLOAT<float, float>(src, kernel, anchor, x, y, c);
          break;
        case CV_64F:
          value = applyMask_FLOAT<float, double>(src, kernel, anchor, x, y, c);
          break;
        }
        break;
     case CV_64F:
      switch(src_depth){
        case 0:
          value = applyMask_FLOAT<double, uchar>(src, kernel, anchor, x, y, c);
          break;
        case 1:
          value = applyMask_FLOAT<double, char>(src, kernel, anchor, x, y, c);
          break;
        case 2:
          value = applyMask_FLOAT<double, unsigned short>(src, kernel, anchor, x, y, c);
          break;
        case 3:
          value = applyMask_FLOAT<double, short>(src, kernel, anchor, x, y, c);
          break;
        case CV_32F:
          value = applyMask_FLOAT<double, float>(src, kernel, anchor, x, y, c);
          break;
        case CV_64F:
          value = applyMask_FLOAT<double, double>(src, kernel, anchor, x, y, c);
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
    case CV_32F:
      dstMat.at<float>(y, x*num_channels + c) = cv::saturate_cast<float>(value);
      break;
    case CV_64F:
      dstMat.at<double>(y, x*num_channels + c) = value;
      break;
  }
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
          if(ddepth < 4 && kernel_depth < 4)
            applyMaskIntWarper(dstMat, src, kernel, 
                ddepth, y, x, c, anchor);
          else
            applyMaskFloatWarper(dstMat, src, kernel,
                ddepth, y, x, c, anchor);
        } 
}

//-------------------------------------------------------
// Sobel part
//-------------------------------------------------------
void utils::createSobelFilter(cv::OutputArray kernel, uchar angle, int depth){
  kernel.create(3, 3, depth);
  std::array<int, 9> data;
  switch(angle){
    case 90: //grad_y
      data = {-1, -2, -1,
              0, 0, 0,
              1, 2, 1};
      break;
    case 0: //grad_x
      data = {-1, 0, 1,
              -2, 0, 2,
              -1, 0, 1};
      break;
    case 45:
      data = {-2, -1, 0,
              -1, 0,  1,
               0, 1,  2};
      break;
    case 135:
      data = { 0, -1, -2,
               1,  0, -1,
               2,  1,  0};
      break;
  }
  for(int y = 0; y < kernel.rows(); y++)
    for(int x = 0; x < kernel.cols(); x++)
      switch(depth){
        case CV_8S:
          kernel.getMat().at<char>(y, x) = data[y*3 + x];
          break;
        case CV_32F:
          kernel.getMat().at<float>(y, x) = data[y*3 + x];
          break;
        case CV_64F:
          kernel.getMat().at<double>(y, x) = data[y*3 + x];
          break;
      }
}

void utils::detectBySobel(cv::InputArray src, cv::OutputArray dst, cv::OutputArray grad_x, cv::OutputArray grad_y){
  CV_Assert(src.type() == CV_8U);
  dst.createSameSize(src, CV_16S);

  // Create filter
  cv::Mat kernel_grad_x, kernel_grad_y;
  createSobelFilter(kernel_grad_x, 0);
  createSobelFilter(kernel_grad_y, 90);

  // Apply filter
  //cv::Mat grad_x, grad_y;
  applyFilter(src, grad_x, CV_16S, kernel_grad_x);
  applyFilter(src, grad_y, CV_16S, kernel_grad_y);

  getL2(grad_x, grad_y, dst);
}

//------------------------------------
// Gaussian filter
//------------------------------------
double gaussFunction(int x, int y, double sigma, int x0, int y0){
  double temp_x = (x - x0) / sigma;
  temp_x = (temp_x / 2) * temp_x;

  double temp_y = (y - y0) / sigma;
  temp_y = (temp_y / 2) * temp_y;

  return exp(-temp_x - temp_y);

}
void createGaussianFilter(cv::OutputArray kernel, uchar ksize, int type=CV_32F, double sigma=-1);
void createGaussianFilter(cv::OutputArray kernel, uchar ksize, int type, double sigma){
  // get default value of sigma
  if(sigma <= 0)
    sigma = 0.3*((ksize - 1) * 0.5 - 1) + 0.8;

  // allocate memory if needed
  kernel.create(ksize, ksize, type);
  cv::Mat kernel_mat = kernel.getMat();

  // Get center
  int y0, x0;
  x0 = y0 = ksize / 2;

  double sum = 0;
  // Get value
  if(type == CV_32F){
    for(int y = 0; y < kernel.rows(); y++)
      for(int x = 0; x < kernel.cols(); x++){
        double value = gaussFunction(x, y, sigma, x0, y0);
        kernel_mat.at<float>(y, x) = value;
        sum += value;
      }
    for(int y = 0; y < kernel.rows(); y++)
      for(int x = 0; x < kernel.cols(); x++){
        kernel_mat.at<float>(y, x) /= sum;
      }
    }
  else if(type == CV_64F){
    for(int y = 0; y < kernel.rows(); y++)
      for(int x = 0; x < kernel.cols(); x++){
        double value = gaussFunction(x, y, sigma, x0, y0);
        kernel_mat.at<double>(y, x) = value;
        sum += value;
      }
    for(int y = 0; y < kernel.rows(); y++)
      for(int x = 0; x < kernel.cols(); x++){
        kernel_mat.at<double>(y, x) /= sum;
      }
  }
}

void utils::applyGaussianFilter(cv::InputArray src,
    cv::OutputArray dst,
    uchar ksize,
    double sigma,
    int ddepth
){

  cv::Mat kernel;
  createGaussianFilter(kernel, ksize, CV_64F, sigma);

  cv::Mat temp_dst;
  applyFilter(src, temp_dst, CV_64F, kernel);

  if(ddepth != CV_64F)
    temp_dst.convertTo(temp_dst, ddepth);
  dst.assign(temp_dst);
}
