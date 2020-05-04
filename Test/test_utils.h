/* Test utils.
 * All function here do nothing to input,
 * Dont have output
 *
 * Only using assert to checking if the input
 * statisfied some condition
 *
 */
#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include <opencv2/core.hpp>
namespace test_case{

// Test if two matrix is same size
void testSameSize(cv::InputArray mat1, cv::InputArray mat2);

// Test if matrix only contains const value
void testIsFilledWithConst(cv::InputArray mat, const uchar value);

// Test if two matrix is the same
void testSameMatrix(cv::InputArray mat1, cv::InputArray mat2);
};

#endif //TEST_UTILS_H_
