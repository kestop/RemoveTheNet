#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\imgcodecs\imgcodecs.hpp>
#include <cmath>

using namespace std;
using namespace cv;

Mat src, dst, cdst;
Mat p_Hough,s_Hough;
string imgName("./pics/WSCCAM8.png");

const int trackBarMax = 150;
const int thresholdMin = 50;
int trackbarValue = trackBarMax / 2;

void Probabilistic_HoughLine(int, void *);
void Standatd_HoughLine(int, void*);
int main()
{

    src = imread(imgName, IMREAD_COLOR);
    cvtColor(src, dst, CV_BGR2GRAY);
    Canny(dst, cdst, 50, 200, 3);

    char threshLabel[50];
    sprintf_s(threshLabel, "T: %d + ", thresholdMin);
    
    namedWindow(imgName+"_HTP", WINDOW_AUTOSIZE);
#if 1
    createTrackbar(threshLabel, imgName, &trackbarValue, trackBarMax, Probabilistic_HoughLine);
    Probabilistic_HoughLine(0, 0);
#else
    createTrackbar(threshLabel, imgName, &trackbarValue, trackBarMax, Standatd_HoughLine);
    Standatd_HoughLine(0, 0);
#endif
    waitKey();
    return 0;
}

void Probabilistic_HoughLine(int, void*)
{
    vector<Vec4i> lines;
    cvtColor(cdst, p_Hough, CV_GRAY2BGR);

    HoughLinesP(cdst, lines, 1, CV_PI / 180, thresholdMin + trackbarValue, 50, 10);
    for each (Vec4i var in lines)
    {
        cout << var << endl;
    }
    cout << "**************************\n";
    for (size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        int x0 = l[0], y0 = l[1], x1 = l[2], y1 = l[3];
        if (atan((y1 - y0) / (x1 - x0)) < CV_PI / 180 * 10 || atan((y1 - y0) / (x1 - x0) > CV_PI / 180 * 170))
        {
            line(p_Hough, Point(x0, y0), Point(x1, y1), Scalar(255, 0, 0), 1, CV_AA);
        }
    }

    imshow(imgName, p_Hough);
    imwrite(imgName + "_HTP", p_Hough);
}
void Standatd_HoughLine(int, void*)
{
    vector<Vec2f> lines;
    cvtColor(cdst, s_Hough, COLOR_GRAY2BGR);
    HoughLines(cdst, lines, 1, CV_PI / 180, thresholdMin + trackbarValue, 0, 0);
    //It gives you as result a vector of couples (\theta, r_{\theta})
    for each (Vec2f var in lines)
    {
        cout << var << endl;
    }
    cout << "******************************\n";

    int alpha = 1000;//the length of line drawn in the image
    for (size_t i = 0; i < lines.size(); i++)
    {
            float rho = lines[i][0], theta = lines[i][1];
        if (theta > CV_PI / 180 * 170 || theta < CV_PI / 180 * 10)
        {
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + alpha * (-b));
            pt1.y = cvRound(y0 + alpha * (a));
            pt2.x = cvRound(x0 - alpha * (-b));
            pt2.y = cvRound(y0 - alpha * (a));
            line(s_Hough, pt1, pt2, Scalar(255, 0, 255), 1, CV_AA);
        }
    }

    imshow(imgName, s_Hough);

}
