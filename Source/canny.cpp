#include "canny.h"
#include "utils.h"
#include <queue>

template <typename T>
inline T getValueFromMat_(cv::InputArray input, int x, int y, int c) {
    if (x < 0 || x >= input.cols())
        return 0;
    if (y < 0 || y >= input.rows())
        return 0;

    int n_channel = input.channels();
    return input.getMat().at<T>(y, x * n_channel + c);
}

template <typename T>
void gradientComputation(cv::InputArray dst, cv::OutputArray gradient_x, cv::OutputArray gradient_y)
{
    // waiting
}

template <typename T>
void getEdgeDirection(cv::InputArray gradient_x, cv::InputArray gradient_y, cv::OutputArray theta)
{
    cv::phase(gradient_x, gradient_y, theta, true); // angle In Degrees
    for (int col = 0; col < theta.cols; ++col)
        for (int row = 0; row < theta.rows; ++row)
        {
            value = theta.at<T>(row, col);

            while (value < 0) value += 180;
            if (value >= 180)  value = value % 180;

            if (value <= 22.5 || value > 157.5)
            {// group of angle 0
                theta.at<T>(row, col) = 0;
            }
            else
            if (value <= 67.5 && value > 22.5)
            {// group of angle 45
                theta.at<T>(row, col) = 45;
            }
            else
            if (value <= 112.5 && value > 67.5)
            {// group of angle 90
                theta.at<T>(row, col) = 90;
            }
            else
            if (value <= 157.5 && value > 112.5)
            {// group of angle 135
                theta.at<T>(row, col) = 135;
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
        neighbor1 = getValueFromMat_(intensity, col - 1, row, -1);
        neighbor2 = getValueFromMat_(intensity, col + 1, row, -1);
        break;
    case 45:
        neighbor1 = getValueFromMat_(intensity, col - 1, row + 1, -1);
        neighbor2 = getValueFromMat_(intensity, col + 1, row - 1, -1);
        break;
    case 90:
        neighbor1 = getValueFromMat_(intensity, col, row - 1, -1);
        neighbor2 = getValueFromMat_(intensity, col , row + 1, -1);
        break;
    case 135:
        neighbor1 = getValueFromMat_(intensity, col - 1, row - 1, -1);
        neighbor2 = getValueFromMat_(intensity, col + 1, row + 1, -1);
    }

    return (intensity.at<T>(row, col) < neighbor1 || intensity.at<T>(row, col) < neighbor2)? 0 : intensity.at<T>(row, col);
}

template <typename T>
void nonMaximumSuppression(cv::InputArray gradient_x, cv::InputArray gradient_y, cv::InputArray theta, cv::OutputArray intensity)
{
    intensity = cv::sqrt(gradient_x.mul(gradient_x) + gradient_y.mul(gradient_y));

    for (int col = 0; col < theta.cols; ++col)
        for (int row = 0; row < theta.rows; ++row)
        {
            intensity.at<T>(row, col) = suppressNonMaxima(intensity, row, col, theta.at<T>(row, col));
        }
}

template <typename T>
void linkEdge(cv::InputArray intensity, cv::InputArray theta, cv::OutputArray dst, std::queue <cv::Point> queue, int low_thres)
{
    cv::Point current_point, neighbor1, neighbor2;
    int row, col;

    while (!queue.empty())
    {
        current_point = queue.front();
        queue.pop();
        row = current_point.y;
        col = current_point.x;

        switch (theta.at<T>(row, col))
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
        if (dst.at<T>(neighbor1.y, neighbor1.x) == 0 && getValueFromMat_(intensity, neighbor1.x, neighbor1.y) >= low_thres)
        {
            dst.at<T>(neighbor1.y, neighbor1.x) == 255;
            queue.push(neighbor1);
        }
        if (dst.at<T>(neighbor2.y, neighbor2.x) == 0 && getValueFromMat_(intensity, neighbor2.x, neighbor2.y) >= low_thres)
        {
            dst.at<T>(neighbor2.y, neighbor2.x) == 255;
            queue.push(neighbor2);
        }
    }
}
template <typename T>
void hysteresis(cv::InputArray intensity, cv::InputArray theta, cv::OutputArray dst, int low_thres, int high_thres)
{
    std::queue <cv::Point> queue_point;
    dst = cv::Scalar::all(0);

    for (int col = 0; col < theta.cols; ++col)
        for (int row = 0; row < theta.rows; ++row)
        {
            if (dst.at<T>(row, col) == 0 && intensity.at<T>(row, col) >= high_thres)
            {
                dst.at<T>(row, col) = 255;
                queue_point.push(cv::Point(col, row));
                linkEdge(intensity, theta, dst, queue_point, low_thres);
            }
        }
}


/*
----------------------MAIN FUNCTION------------------------------
*/
void utils::detectByCanny(cv::InputArray src, cv::OutputArray dst, int low_thres, int high_thres)
{
    printf("hello world!");

    // Image smoothing
    utils::applyGaussianFilter(src, dst, 3); // can change kernel_size

    // Gradient computation
    cv::Mat gradient_x, gradient_y;
    //cv::Mat gradient_x(cv::Size(src.cols, src.rows), src.type(), cv::Scalar::all(0)), 
	//        gradient_y(cv::Size(src.cols, src.rows), src.type(), cv::Scalar::all(0));
    //gradientComputation(dst, gradient_x, gradient_y);

    // Edge direction computation
    cv::Mat theta;
    //cv::Mat theta(cv::Size(src.cols, src.rows), src.type(), cv::Scalar::all(0));
    //getEdgeDirection(gradient_x, gradient_y, theta);

    // Non-maximum suppresion
    cv::Mat intensity;
    //cv::Mat intensity(cv::Size(src.cols, src.rows), src.type(), cv::Scalar::all(0));
    //nonMaximumSuppression(gradient_x, gradient_y, theta, intensity);

    // Hysteresis
    //hysteresis(intensity, theta, dst, low_thres, high_thres);
}