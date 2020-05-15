#include "canny.h"
#include "utils.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <queue>

#define PI 3.14159265

/*
----------------------TEMPLATE HELPER FUNCTION SECTION------------------------------
*/


/* Check if Point p is a valid point in matrix input */
bool isValidPoint(cv::InputArray input, cv::Point p)
{
    if (p.x < 0 || p.x >= input.cols())
        return false;
    if (p.y < 0 || p.y >= input.rows())
        return false;
    return true;
}

/* get  value of input[row, col]  */
template <typename T>
inline T getValueFromMat_(cv::InputArray input, int col, int row) {
    if (col < 0 || col >= input.cols())
        return 0;
    if (row < 0 || row >= input.rows())
        return 0;

    return input.getMat().at<T>(row, col);
}

/* calc gradient of src, output is x-gradient, y-gradient and dst as gradient magnitude */
/* input src: must be CV_8U */
/* output dst, gradient_x, gradient_y: will be CV_16S */
template <typename T>
void gradientComputation(cv::InputArray src, cv::OutputArray dst, cv::OutputArray gradient_x, cv::OutputArray gradient_y)
{
    utils::detectBySobel(src, dst, gradient_x, gradient_y);
}

/* calc gradient orientation from x-gradient, y-gradient */
/* input: gradient_x, gradient_y: must be CV_16S */
/* output: theta: will be CV_16S */
template <typename T>
void getEdgeDirection(cv::InputArray gradient_x, cv::InputArray gradient_y, cv::OutputArray theta)
{
    theta.create(gradient_x.getMat().size(), gradient_x.getMat().type());

    T x, y;
    for (int col = 0, value; col < theta.cols(); ++col)
        for (int row = 0; row < theta.rows(); ++row)
        {
            x = getValueFromMat_<T>(gradient_x, col, row);
            y = getValueFromMat_<T>(gradient_y, col, row);

            value = atan2(y, x) * 180 / PI;

            while (value < 0) value += 180;
            while (value >= 180)  value -= 180;


            if (value <= 22.5 || value > 157.5)
            {// group of angle 0
                theta.getMat().at<T>(row, col) = 0;
            }
            else
                if (value <= 67.5 && value > 22.5)
                {// group of angle 45
                    theta.getMat().at<T>(row, col) = 45;
                }
                else
                    if (value <= 112.5 && value > 67.5)
                    {// group of angle 90
                        theta.getMat().at<T>(row, col) = 90;
                    }
                    else
                        if (value <= 157.5 && value > 112.5)
                        {// group of angle 135
                            theta.getMat().at<T>(row, col) = 135;
                        }
        }
}

/* help function of nonMaximumSuppression function */
template <typename T>
T suppressNonMaxima(cv::InputArray intensity, int row, int col, T angle, int size_kernel)
{
    T neighbor1, neighbor2, max_value = intensity.getMat().at<T>(row, col);

    size_kernel = (size_kernel <= 2) ? 3 : size_kernel;

    for (int i = 1; i <= size_kernel/2; ++i)
    {
        switch (angle)
        {
        case 0:
            neighbor1 = getValueFromMat_<T>(intensity, col - i, row);
            neighbor2 = getValueFromMat_<T>(intensity, col + i, row);
            break;
        case 45:
            neighbor1 = getValueFromMat_<T>(intensity, col - i, row - i);
            neighbor2 = getValueFromMat_<T>(intensity, col + i, row + i);
            break;
        case 90:
            neighbor1 = getValueFromMat_<T>(intensity, col, row - i);
            neighbor2 = getValueFromMat_<T>(intensity, col, row + i);
            break;
        case 135:
            neighbor1 = getValueFromMat_<T>(intensity, col - i, row + i);
            neighbor2 = getValueFromMat_<T>(intensity, col + i, row - i);
        }

        max_value = (neighbor1 > max_value) ? neighbor1 : max_value;
        max_value = (neighbor2 > max_value) ? neighbor2 : max_value;
    }

    // intensity[row, col] be maintained or get 0-value whether it is maxima by orientation or not
    return (intensity.getMat().at<T>(row, col) < max_value) ? 0 : intensity.getMat().at<T>(row, col);
}

/* get theta and intensity, output is intensity after suppressing non-maxima */
template <typename T>
void nonMaximumSuppression(cv::InputArray theta, cv::OutputArray intensity, int size_kernel)
{
    for (int col = 0; col < theta.cols(); ++col)
        for (int row = 0; row < theta.rows(); ++row)
        {
            intensity.getMat().at<T>(row, col) = suppressNonMaxima(intensity, row, col, theta.getMat().at<T>(row, col), size_kernel);
        }
}

/* help function of hysteresis function */
template <typename T>
void linkEdge(cv::InputArray intensity, cv::InputArray theta, cv::OutputArray dst, std::queue <cv::Point>* queue, int low_thres)
{
    cv::Point current_point, neighbor1, neighbor2;
    int row, col;

    while (!queue->empty())
    {
        current_point = queue->front();
        queue->pop();
        row = current_point.y;
        col = current_point.x;

        switch (theta.getMat().at<T>(row, col))
        {
        case 0:
            neighbor1 = cv::Point(col, row - 1);
            neighbor2 = cv::Point(col, row + 1);
            break;
        case 45:
            neighbor1 = cv::Point(col - 1, row + 1);
            neighbor2 = cv::Point(col + 1, row - 1);
            break;
        case 90:
            neighbor1 = cv::Point(col - 1, row);
            neighbor2 = cv::Point(col + 1, row);
            break;
        case 135:
            neighbor1 = cv::Point(col - 1, row - 1);
            neighbor2 = cv::Point(col + 1, row + 1);
        }

        // link point
        if (isValidPoint(dst, neighbor1) && dst.getMat().at<T>(neighbor1.y, neighbor1.x) == 0
            && getValueFromMat_<T>(intensity, neighbor1.x, neighbor1.y) >= low_thres)
        {
            dst.getMat().at<T>(neighbor1.y, neighbor1.x) = 255;
            queue->push(neighbor1);
        }
        if (isValidPoint(dst, neighbor2) && dst.getMat().at<T>(neighbor2.y, neighbor2.x) == 0
            && getValueFromMat_<T>(intensity, neighbor2.x, neighbor2.y) >= low_thres)
        {
            dst.getMat().at<T>(neighbor2.y, neighbor2.x) = 255;
            queue->push(neighbor2);
        }
    }
}

/* Link the pixels of the edge together */
template <typename T>
void hysteresis(cv::InputArray intensity, cv::InputArray theta, cv::OutputArray dst, int low_thres, int high_thres)
{
    std::queue <cv::Point> queue_point;

    dst.create(intensity.size(), intensity.type());
    cv::Mat temp(intensity.size(), intensity.type(), cv::Scalar::all(0));
    dst.assign(temp);

    for (int col = 0; col < theta.cols(); ++col)
        for (int row = 0; row < theta.rows(); ++row)
        {
            if (dst.getMat().at<T>(row, col) == 0 && intensity.getMat().at<T>(row, col) >= high_thres)
            {
                dst.getMat().at<T>(row, col) = 255;
                queue_point.push(cv::Point(col, row));
                linkEdge<T>(intensity, theta, dst, &queue_point, low_thres);
            }
        }
}


/*
----------------------MAIN FUNCTION------------------------------
*/
void utils::detectByCanny(cv::InputArray src, cv::OutputArray dst, int low_thres, int high_thres, int size_kernel_gauss, int size_kernel_nms)
{
    cv::Mat src_input = src.getMat();
    src_input.convertTo(src_input, CV_8U);

    // Image smoothing
    cv::Mat smoothImage;
    utils::applyGaussianFilter(src_input, smoothImage, size_kernel_gauss); // can change kernel_size

    // Gradient computation
    cv::Mat gradient_x, gradient_y;
    cv::Mat intensity;

    gradientComputation<uchar>(smoothImage, intensity, gradient_x, gradient_y);

    // Edge direction computation
    cv::Mat theta;
    getEdgeDirection<int16_t>(gradient_x, gradient_y, theta);

    // Non-maximum suppresion
    nonMaximumSuppression<int16_t>(theta, intensity, size_kernel_nms);

    theta.convertTo(theta, CV_8U);
    intensity.convertTo(intensity, CV_8U);
    
    // Hysteresis
    hysteresis<uchar>(intensity, theta, dst, low_thres, high_thres);
}
