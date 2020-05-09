#ifndef CANNY_H_
#define CANNY_H_

#include <opencv2/core.hpp>

namespace utils{
  /* Canny edge detection method
   * 
   * @param src: Input Array. Should be CV_8U
   * @param dst: Output Array. Will be CV_16S
   * @param low_thres: Low Threshold
   * @param high_thres: High Threshold
   *
   */ 
  void detectByCanny(cv::InputArray src, cv::OutputArray dst,
      int low_thres, int high_thres);

};


#endif //CANNY_H_

