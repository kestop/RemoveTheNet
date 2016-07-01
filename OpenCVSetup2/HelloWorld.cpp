/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - HelloOpenCV.cpp
// TOPIC: basic image operations
//
// Getting-Started-File 
// University of Bristol
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include <opencv2\opencv.hpp>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>

/////////////////////////////////////////////////////////////////////////////
//
// easy to use (but slow) pixel manipulation
//
/////////////////////////////////////////////////////////////////////////////

//returns value between 0 and 255 of pixel at image position (x,y)
unsigned char _getPixel(IplImage* image,    //the image 
    int x,              //abscissa
    int y               //ordinate
    )
{
    return ((unsigned char*)(image->imageData + image->widthStep*y))[x*image->nChannels];
}

//sets pixel at image position (x,y)
void _setPixel(IplImage* image,     //the image
    int x,               //abscissa
    int y,               //ordinate
    unsigned char value  //new pixel value
    )
{
    ((unsigned char*)(image->imageData + image->widthStep*y))[x*image->nChannels] = value;
}

/////////////////////////////////////////////////////////////////////////////
//
// main procedure
//
/////////////////////////////////////////////////////////////////////////////

// main procedure: defines the entry point of your application
int main(void)
{
    //create gray scale image at 400x300 pixels
    IplImage* myImage = cvCreateImage(cvSize(400, 300), IPL_DEPTH_8U, 1);
    printf("Image created...\n");

    //set image to black
    cvZero(myImage);
    printf("Image cleared...\n");

    //set pixel at (x=200,y=100) to 255
    _setPixel(myImage, 200, 100, 255);
    printf("Pixel set...\n");

    //read this pixel value
    unsigned char value = _getPixel(myImage, 200, 100);
    printf("Pixel read...\n");

    //draw a line
    cvLine(myImage, cvPoint(20, 200), cvPoint(380, 200), cvScalar(155), 1);
    printf("Line drawn...\n");

    //print "HelloOpenCV"
    CvFont font;
    cvInitFont(&font, CV_FONT_VECTOR0, 1.5, 0.8, 0.0, 1);
    cvPutText(myImage, "Hello OpenCV", cvPoint(50, 180), &font, cvScalar(255));
    printf("Hello OpenCV printed in image...\n");

    //save the image
    cvSaveImage("HelloOpenCV.jpg", myImage);
    printf("Image saved to file...\n");

    //show the image
    cvNamedWindow("ImageWindow", CV_WINDOW_AUTOSIZE);
    cvShowImage("ImageWindow", myImage);
    printf("Image displayed...\n");

    //wait till key is pressed
    printf("PRESS A KEY NOW...\n");
    cvWaitKey();

    //release the image from memory
    cvReleaseImage(&myImage);
    printf("Image released...\n");
    return 0;
}
