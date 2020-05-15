#ifndef CANNY_H_
#define CANNY_H_

#include <opencv2/core.hpp>

namespace utils{
  /* Canny edge detection method
   * 
   * @param src: Input Array. Should be CV_8U
   * @param dst: Output Array. Will be CV_8U
   * @param low_thres: Low Threshold
   * @param high_thres: High Threshold
   * @param size_kernel_gauss: GaussianFilter's kernel size. Default value = 3
   * @param size_kernel_nms: NonMaxSupression's kernel size. Default value = 5
   *
   */ 
  void detectByCanny(cv::InputArray src, cv::OutputArray dst,
      int low_thres, int high_thres, 
      int size_kernel_gauss = 3, int size_kernel_nms = 5);

};


#endif //CANNY_H_

