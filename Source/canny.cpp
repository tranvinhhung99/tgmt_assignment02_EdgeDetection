#include "canny.h"
#include "utils.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <queue>

#define PI 3.14159265

bool isValidPoint(cv::InputArray input, cv::Point p)
{
    if (p.x < 0 || p.x >= input.cols())
        return 0;
    if (p.y < 0 || p.y >= input.rows())
        return 0;
    return true;
}

template <typename T>
inline T getValueFromMat_(cv::InputArray input, int x, int y) {
    if (x < 0 || x >= input.cols())
        return 0;
    if (y < 0 || y >= input.rows())
        return 0;

    return input.getMat().at<T>(y, x);
}

template <typename T>
void gradientComputation(cv::InputArray src, cv::OutputArray dst, cv::OutputArray gradient_x, cv::OutputArray gradient_y)
{
    utils::detectBySobel(src, dst, gradient_x, gradient_y);
}

template <typename T>
void getEdgeDirection(cv::InputArray gradient_x, cv::InputArray gradient_y, cv::OutputArray theta)
{
    //cv::phase(gradient_x, gradient_y, theta, true); // angle In Degrees
    theta.create(gradient_x.getMat().size(), gradient_x.getMat().type());

    T x, y;
    for (int col = 0, value; col < theta.cols(); ++col)
        for (int row = 0; row < theta.rows(); ++row)
        {
            x = getValueFromMat_<T>(gradient_x, row, col);
            y = getValueFromMat_<T>(gradient_y, row, col);
            value = atan2(y,  x) * 180 / PI;

            while (value < 0) value += 180;
            if (value >= 180)  value = value % 180;

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
            else
            {
                printf("What's wrong in getEdgeDirection!!!\n");
            }
        }
}

template <typename T>
T suppressNonMaxima(cv::InputArray intensity, int row, int col, T angle)
{
    T neighbor1, neighbor2;
    switch (angle)
    {
    case 0:
        neighbor1 = getValueFromMat_<T>(intensity, col - 1, row);
        neighbor2 = getValueFromMat_<T>(intensity, col + 1, row);
        break;
    case 45:
        neighbor1 = getValueFromMat_<T>(intensity, col - 1, row + 1);
        neighbor2 = getValueFromMat_<T>(intensity, col + 1, row - 1);
        break;
    case 90:
        neighbor1 = getValueFromMat_<T>(intensity, col, row - 1);
        neighbor2 = getValueFromMat_<T>(intensity, col , row + 1);
        break;
    case 135:
        neighbor1 = getValueFromMat_<T>(intensity, col - 1, row - 1);
        neighbor2 = getValueFromMat_<T>(intensity, col + 1, row + 1);
    }

    return (intensity.getMat().at<T>(row, col) < neighbor1
        || intensity.getMat().at<T>(row, col) < neighbor2)? 0 : intensity.getMat().at<T>(row, col);
}

template <typename T>
void nonMaximumSuppression(cv::InputArray theta, cv::OutputArray intensity)
{
    for (int col = 0; col < theta.cols(); ++col)
        for (int row = 0; row < theta.rows(); ++row)
        {
            intensity.getMat().at<T>(row, col) = suppressNonMaxima(intensity, row, col, theta.getMat().at<T>(row, col));
        }
}

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
            neighbor1 = cv::Point(col - 1, row - 1);
            neighbor2 = cv::Point(col + 1, row + 1);
            break;
        case 90:
            neighbor1 = cv::Point(col - 1, row);
            neighbor2 = cv::Point(col + 1, row);
            break;
        case 135:
            neighbor1 = cv::Point(col - 1, row + 1);
            neighbor2 = cv::Point(col + 1, row - 1);
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
void utils::detectByCanny(cv::InputArray src, cv::OutputArray dst, int low_thres, int high_thres)
{
    printf("Hello World!\n");

    cv::Mat src_input = src.getMat();
    src_input.convertTo(src_input, CV_8U);

    // Image smoothing
    cv::Mat smoothImage;
    utils::applyGaussianFilter(src_input, smoothImage, 3); // can change kernel_size
    //cv::imshow("applyGaussianFilter", smoothImage); // debug


    // Gradient computation
    cv::Mat gradient_x, gradient_y;
    cv::Mat intensity;
    gradientComputation<uchar>(smoothImage, intensity, gradient_x, gradient_y);
    //cv::imshow("gradient_x", gradient_x); // debug
    //cv::imshow("gradient_y", gradient_y); // debug

    // Edge direction computation
    cv::Mat theta;
    getEdgeDirection<int16_t>(gradient_x, gradient_y, theta);
    
    theta.convertTo(theta, CV_8U);
    cv::imshow("theta", theta); // debug
    intensity.convertTo(intensity, CV_8U);
    cv::imshow("intensity", intensity); // debug

    // Non-maximum suppresion
    nonMaximumSuppression<uchar>(theta, intensity);
    cv::imshow("intensity", intensity); // debug

    // Hysteresis
    hysteresis<uchar>(intensity, theta, dst, low_thres, high_thres);
}
