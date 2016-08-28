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
const string videoName("./videos/ActionIndoorSportsBristolCam2.mp4");
const string outputName("./output/sample4.avi");
const string wndName("Image");
const string wndName2("Mask");


static void onMouse(int event, int x, int y, int, void*);
static void onMouseForMaskWnd(int event, int x, int y, int, void*);
void hardPriorPts();
static double angle(Point p1, Point p2);
static double length(Point p1, Point p2);
void defineFourEdges(vector<double> angles, double lineLength);
bool ifAtCenter(Point p1, Point p2, const Mat & image);
void findNet(const Mat& image, vector<double> angles, double lineLength);
void makeMask(const vector<Vec4i> lines,
    const vector<double> angles, const double lineLength);
void drawSquare(Mat& src, Mat& dst);
void help();
void eventLoop();
void templateMatching();
void drawTheNet();
void sortVector(vector<vector<Point>>& pp, int xORy);
void inpaintWithMask();
void processVideo(const int startFrameIndex);
void Erosion(Mat& src, Mat& erosion_dst);
void Dilation(Mat& src, Mat& dilation_dst);
void capAFrameFromVideo(const int frameIndex);
void maskDisplay();
void findLines(int, void*);
void quadraticFit(vector<Point>& p);
void linearFit(const vector<Point>& p, double & a, double & b);

void generateTheNet();
int getGridPointsNumber(double l1, double l2, double ld);
void getDiff(int c0, int c1, int c2, int c3, int n, double &d, double &ds);
void tideUpJtss(vector<vector<Point>> &pp);
void predictGapJoints(vector<vector<Point>> &pp);
void initJtss(const vector<vector<Point>>& pp, vector<vector<Point>>& Jtss);
void extendTopOrLeft(vector<vector<Point>>& pp, vector<vector<Point>>& Jtss,
    double dx, double dy, size_t i, size_t sz);
void extendBottomOrRight(vector<vector<Point>> &pp, vector<vector<Point>> &Jtss,
    double dx, double dy, size_t i, size_t sz);
void expandJointsToWholeNet(vector<vector<Point>> &pp, vector<vector<Point>> &Jtss);
bool outOfImage(Point p);
double getTheD(vector<Point> p, int xORy);
void predictGapPointsNum(const vector<cv::Point> pts, vector<double>& gapJts);
void avgGapJoints(const vector<vector<double>> &gJ, double fGJ[]);
void generateGapJoints(double fGJ[], vector<Point> &pts);
bool nextLine(Point seed);
