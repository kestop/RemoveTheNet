#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\opencv.hpp>
#include "mandrillRestore1.h"

using namespace std;
using namespace cv;

Mat restore0(Mat M);

Mat restore1(Mat M);

Mat restore2(Mat M);

Mat restore3(Mat M);

Mat restore4(Mat M);

Mat convolution(Mat Input, int  Kernel[3][3]);

Mat convolution2(Mat M, int kernel[3][3]);


int main(int argc, char* argv[])
{
    string imageName0("mandrillRGB.jpg");
    string imageName1("mandrill0.jpg");
    string imageName2("mandrill1.jpg");
    string imageName3("mandrill2.jpg");
    string imageName4("mandrill3.jpg");
    string imageName5("mandrill4.jpg");
    string imageName6("mandrill.jpg");

    int kernel[3][3] = {
        { 1 / 9, 1 / 9, 1 / 9 } ,   /*  initializers for row indexed by 0 */
        { 1 / 9, 1 / 9, 1 / 9 } ,   /*  initializers for row indexed by 1 */
        { 1 / 9, 1 / 9, 1 / 9 }   /*  initializers for row indexed by 2 */
    };


    Mat m0;
    Mat m1, m2, m3, m4, m5, m6;


    m0 = imread(imageName0, IMREAD_COLOR);
    m1 = imread(imageName1, IMREAD_COLOR);
    m2 = imread(imageName2, IMREAD_COLOR);
    m3 = imread(imageName3, IMREAD_COLOR);
    m4 = imread(imageName4, IMREAD_COLOR);
    m5 = imread(imageName5, IMREAD_COLOR);
    m6 = imread(imageName6, IMREAD_GRAYSCALE);


    if (m0.empty() || m1.empty() || m2.empty() || m3.empty()
        || m4.empty() || m5.empty() || m6.empty())
    {
        cout << "images failed to read." << endl;
    }

    /*cout << "intensity in 0 " << m0.at<Vec3b>(248, 236) << endl;
    cout << "intensity in test2 " << m4.at<Vec3b>(248, 236) << endl;
    cout << "intensity in 0 " << m0.at<Vec3b>(53, 330) << endl;
    cout << "intensity in tst " << m4.at<Vec3b>(53, 330) << endl;
    cout << "intensity in 0 bright pixel in b channel " << m0.at<Vec3b>(238, 248) << endl;
    cout << "intensity in test3 " << m4.at<Vec3b>(238, 248) << endl;*/
    cout << kernel[1][1] << "<-kernel" << endl;
    /* Mat m11 = restore0(m1);
     Mat m22 = restore1(m2);
     Mat m33 = restore2(m3);*/
     //Mat m44 = restore3(m4);
    //Mat m55 = restore4(m5);
    Mat m66 = convolution2(m6, kernel);

    imwrite("outMandrill.jpg", m66);
    namedWindow(imageName6, WINDOW_AUTOSIZE);
    imshow(imageName6, m66);

    waitKey(0);

    return 0;
}


Mat restore0(Mat M)
{
    // bgr , 2 -> 0, 1 -> 2, 0 -> 1
    //accpet only char type matrices
    CV_Assert(M.depth() == CV_8U);

    int channels = M.channels();
    int nCols = M.cols;
    int nRows = M.rows*channels;

    if (M.isContinuous())
    {
        cout << "Continuous!\n";
        nCols *= nRows;
        nRows = 1;
    }

    uchar tmp1;

    uchar* p;
    for (int i = 0; i < nRows; i++)
    {
        p = M.ptr<uchar>(i);
        for (int j = 0; j + 2 < nCols; j += 3)
        {
            tmp1 = p[j];// 0 -> tmp1
            p[j] = p[j + 2]; // 2 -> 0
            p[j + 2] = p[j + 1]; // 1 -> 2
            p[j + 1] = tmp1; // 0-> 1
        }
    }

    return M;
}

Mat restore1(Mat M)
{
    // r channel move back towards right down by 32 pixels in both x and y direction

    Mat M_clone = M.clone();
    for (int i = 0; i < M.rows; i++)
    {
        for (int j = 0; j < M.cols; j++)
        {
            if (i < 32 || j < 32)
            {
                M.at<Vec3b>(i, j)[2] = 0;
                if (j < 32 && i >= 32)
                {
                    M.at<Vec3b>(i, j)[2] = M_clone.at<Vec3b>(i - 32, j + 480)[2];
                }
            }
            else if (i >= 32 && j >= 32)
            {
                M.at<Vec3b>(i, j)[2] = M_clone.at<Vec3b>(i - 32, j - 32)[2];
            }
        }
    }
    return M;
}

Mat restore2(Mat M)
{
    //reverse every channel in every pixel
    //accpet only char type matrices
    CV_Assert(M.depth() == CV_8U);

    int channels = M.channels();
    int nCols = M.cols;
    int nRows = M.rows*channels;

    if (M.isContinuous())
    {
        cout << "Continuous!\n";
        nCols *= nRows;
        nRows = 1;
    }

    uchar* p;
    for (int i = 0; i < nRows; i++)
    {
        p = M.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++)
        {
            p[j] = 255 - p[j];
        }
    }
    return M;
}

Mat restore3(Mat M)
{
    //try modulo 256 addition
    //cannot recover all info because some has been lost
    uchar addtion = 64;

    for (int i = 0; i < M.rows; i++)
    {
        for (int j = 0; j < M.cols; j++)
        {
            //M.at<Vec3b>(i, j)[0] += addtion;
            //M.at<Vec3b>(i, j)[1] += addtion;
            M.at<Vec3b>(i, j)[2] += addtion;

        }
    }

    return M;
}

Mat restore4(Mat M)
{
    //convert from hsv to bgr space
    Mat M_bgr;
    cvtColor(M, M_bgr, CV_HSV2BGR);


    return M_bgr;
}

Mat convolution(Mat Input, int Kernel[3][3])
{
    // accept only char type matrices
    CV_Assert(Input.depth() != sizeof(uchar));
    

    for (int x = 0; x < Input.rows; x++)
    {
        for (int y = 0; y < Input.cols; y++)
        {
            for (int m = -1; m < 2; m++)
            {
                for (int n = -1; n < 2; n++)
                {
                    if (x - m >= 0 && x - m < Input.rows && y - n >= 0 && y - n < Input.cols)
                    {
                        Input.at<uchar>(x - m, y - n) +=
                            (Input.at<uchar>(x - m, y - n) * Kernel[m + 1][n + 1]);
           

                    }
                }
            }
        }
    }
    return Input;
}

Mat convolution2(Mat M, int kernel[3][3])
{
    
    // find center position of kernel (half of kernel size)
    int kCenterX = 3 / 2;
    int kCenterY = 3 / 2;
    int kRows = 3, kCols = 3;
    int rows = M.rows, cols = M.cols;
    int i, j,m,n,mm,nn,ii,jj;
    Mat out = M.clone();

    for (i = 0; i < rows; ++i)              // rows
    {
        for (j = 0; j < cols; ++j)          // columns
        {
            for (m = 0; m < kRows; ++m)     // kernel rows
            {
                mm = kRows - 1 - m;      // row index of flipped kernel

                for (n = 0; n < kCols; ++n) // kernel columns
                {
                    nn = kCols - 1 - n;  // column index of flipped kernel

                                         // index of input signal, used for checking boundary
                    ii = i + (m - kCenterY);
                    jj = j + (n - kCenterX);

                    // ignore input samples which are out of bound
                    if (ii >= 0 && ii < rows && jj >= 0 && jj < cols)
                        out.at<uchar>(i,j) += M.at<uchar>(ii,jj) * kernel[mm][nn];
                }
            }
        }
    }
    return out;
}
