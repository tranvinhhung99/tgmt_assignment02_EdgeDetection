#ifndef UTILS_H_
#define UTILS_H_

#include<opencv2/core/mat.hpp>

namespace utils{
  const int BORDER_CONSTANT = 0;
  
  /* Re-implementation of cv::filter2D
   * Note: Currently only support 
   * constant padding version. Will update in later
   *
   * @param src: Source input image
   * @param dst: Output destination
   * @param ddepth: Destination depth. -1 for same as src.
   * 
   * Currently support: CV_8U, CV_8S, CV_16U, CV_16S
   *
   * @param kernel
   * @parma anchor: 2D Point. Default use kernel center
   *
   */
  void applyFilter(cv::InputArray src, 
      cv::OutputArray dst, 
      int ddepth,
      cv::InputArray kernel,
      cv::Point anchor=cv::Point(-1, -1)

  );
};

#endif //UTILS_H_
