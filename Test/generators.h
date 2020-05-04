/* Test Generator function
 *
 * - Generate random input for input
 *
 *
 *
 *
 */ 
#ifndef TEST_GENERATOR_H_
#define TEST_GENERATOR_H_

#include <opencv2/core.hpp>

namespace test_case{
  // Generate random matrix with known size and depth
  void generateRandomMatrix(cv::OutputArray, int num_row, int num_col, int ddepth);
  void generateRandomMatrix(cv::OutputArray, cv::Size, int ddepth);


}

#endif //TEST_GENERATOR_H_
