#pragma once
#include <iostream>
#include <string>
#include <cmath>
#include <ctime>
#include<iomanip>

using namespace cv;
using namespace std;


Mat image0, image, mask, sample, result;
Mat gray, edges, pyr;
const string videoName("./videos/WillowsSportsCentreCam8.mp4");
const string outputName("./output/sample4.avi");
const string wndName("Image");
const string wndName2("Mask");

//User interface
static void onMouse(int event, int x, int y, int, void*);
static void onMouseForMaskWnd(int event, int x, int y, int, void*);
void capAFrameFromVideo(const int frameIndex);
void inpaintWithMask();
void drawTheNet();
void processVideo(const int startFrameIndex);
//Event loop
void eventLoop();
//edge-based detection 
void findNet(const Mat& image, vector<double> angles, double lineLength);
void defineFourEdges(vector<double> angles, double lineLength);
void makeMask(const vector<Vec4i> lines,
    const vector<double> angles, const double lineLength);
void generateTheNet();
void getDiff(int c0, int c1, int c2, int c3, int n, double &d, double &ds);
void Erosion(Mat& src, Mat& erosion_dst);
void Dilation(Mat& src, Mat& dilation_dst);
//grid-based detection
void templateMatching();
void sortVector(vector<vector<Point>>& pp, int xORy);
void findLines(int, void*);
void quadraticFit(vector<Point>& p);
void linearFit(const vector<Point>& p, double & a, double & b);
void tideUpJtss(vector<vector<Point>> &pp);
void predictGapJoints(vector<vector<Point>> &pp);
void initJtss(const vector<vector<Point>>& pp, vector<vector<Point>>& Jtss);
void extendTopOrLeft(vector<vector<Point>>& pp, vector<vector<Point>>& Jtss,
    double dx, double dy, size_t i, size_t sz);
void extendBottomOrRight(vector<vector<Point>> &pp, vector<vector<Point>> &Jtss,
    double dx, double dy, size_t i, size_t sz);
void expandJointsToWholeNet(vector<vector<Point>> &pp, vector<vector<Point>> &Jtss);
void predictGapPointsNum(const vector<cv::Point> pts, vector<double>& gapJts);
void avgGapJoints(const vector<vector<double>> &gJ, double fGJ[]);
void generateGapJoints(double fGJ[], vector<Point> &pts);
//helper functions
void maskDisplay();
int getGridPointsNumber(double l1, double l2, double ld);
bool outOfImage(Point p);
double getTheD(vector<Point> p, int xORy);
bool nextLine(Point seed);
static double angle(Point p1, Point p2);
static double length(Point p1, Point p2);
void hardPriorPts();
void help();
bool ifAtCenter(Point p1, Point p2, const Mat & image);

