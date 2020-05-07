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

  /* Create 3x3 Sobel Filter from parameters.
   *
   * @param kernel: Output kernel
   * @param angle: Can only be 0, 45, 90, 135
   * @param depth: Currently only support CV_8S
   *
   */ 
  void createSobelFilter(cv::OutputArray kernel, unsigned char angle, int depth=CV_8S);

  /* Create 3x3 Prewitt Filter from parameters.
   *
   * @param kernel: Output kernel
   * @param angle: Can only be 0, 45, 90, 135
   * @param depth: Currently only support CV_8S
   *
   */ 
  void createPrewittFilter(cv::OutputArray kernel, unsigned char angle, int depth=CV_8S);


  /* Apply Gaussian Filter from parameters.
   *
   * @param src: Input image. Must be CV_32F depth
   * @param dst: Output image
   * @param kernel_size: Kernel size of Gaussian Filter. Must be odd number
   *
   */ 
  void applyGaussianFilter(cv::InputArray src,
      cv::OutputArray dst,
      unsigned char kernel_size
  );


  /* Detect edge by Sobel algorithm
   *
   * @param src: Input image, must be in CV_8U
   * @param dst: Output Image, will output in CV_16S
   *
   */
  void detectBySobel(cv::InputArray src, cv::OutputArray dst);

  /* Detect edge by Prewitt algorithm
   *
   * @param src: Input image, must be in CV_8U
   * @param dst: Output Image, will output in CV_16S
   *
   */
  void detectByPrewitt(cv::InputArray src, cv::OutputArray dst);

  /* Detect edge by Prewitt algorithm
   *
   * @param src: Input image, must be in CV_8U
   * @param dst: Output Image, will output in CV_16S
   *
   */
  void detectByPrewitt(cv::InputArray src, cv::OutputArray dst);
};

#endif //UTILS_H_
