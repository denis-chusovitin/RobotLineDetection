#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include <time.h>
#include <stack>

using namespace cv;
using namespace std;

#define    MAX_LINE_WIDTH 40
#define	   MIN_LINE_WIDTH 10
#define       IMAGE_WIDTH 650
#define      IMAGE_HEIGTH 650
#define		 MAX_DISTANCE 40
#define  MIN_CONTOUR_SIZE 6
#define			EPS_APROX 100

typedef vector<Point> contour;

struct Line
{
	float a;
	float b;
	float c;

	Line(Point p1, Point p2) 
	{
		a = p1.y - p2.y;
		b = p2.x - p1.x;
		c = p1.x * p2.y - p2.x * p1.y;
	}

	float distToLine(Point p)
	{
		return  abs(p.x * a + p.y * b + c);
	}

};


class Graph
{
private:
	vector<vector<bool>> matrix;
	int size;

	int sqr_distance(contour cntr1, contour cntr2)
	{
		int min_distance = (cntr1.at(0).x - cntr2.at(0).x) * (cntr1.at(0).x - cntr2.at(0).x)
			+ (cntr1.at(0).y - cntr2.at(0).y) * (cntr1.at(0).y - cntr2.at(0).y);
		int i = 0, j = 0;

		while (i < cntr1.size())
		{
			while (j < cntr2.size())
			{
				int dist = (cntr1.at(i).x - cntr2.at(j).x) * (cntr1.at(i).x - cntr2.at(j).x)
					+ (cntr1.at(i).y - cntr2.at(j).y) * (cntr1.at(i).y - cntr2.at(j).y);

				if (dist < min_distance)
				{
					min_distance = dist;
				}
				j += 1;
			}
			i += 1;
		}

		return min_distance;
	}

	
public:
	Graph(vector<contour> contours)
	{
		size = contours.size();
		vector<vector<bool>> temp(size, vector<bool>(size, false));

		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				if (sqr_distance(contours[i], contours[j]) < MAX_DISTANCE * MAX_DISTANCE)
				{
					temp[i][j] = true;
					temp[j][i] = true;
				}
			}
		}
		
		matrix = temp;
	}

	void dfs(int vertex, vector<bool>& passed_vertex)
	{
		passed_vertex.at(vertex) = true;

		for (int i = 0; i < size; ++i)
		{
			if (!passed_vertex.at(i) && matrix.at(vertex).at(i))
			{
				dfs(i, passed_vertex);
			}
		}
	}
};

bool is_contains_zeros(contour cntr)
{
	int i = 0;
	bool is_contains = true;

	while (i < cntr.size() && cntr[i].y > 1)
	{
		i++;
	}

	if (i == cntr.size())
	{
		is_contains = false;
	}

	return is_contains;
}

vector<contour> filter_contours(vector<contour>& contours, Graph& graph)
{
	vector<bool> passed_vertex(contours.size(), false);

	for (int i = 0; i < contours.size(); i++)
	{
		if (is_contains_zeros(contours.at(i)) && !passed_vertex.at(i))
		{
			graph.dfs(i, passed_vertex);
		}
	}


	for (int i = 0; i < contours.size(); i++)
	{
		if (contours.at(i).size() < MIN_CONTOUR_SIZE && passed_vertex.at(i))
		{
			graph.dfs(i, passed_vertex);
		}
	}
	
	for (int i = 0; i < contours.size(); i++)
	{
		if (contours.at(i).size() < MIN_CONTOUR_SIZE)
		{
			passed_vertex[i] = true;
		}
	}
	
	vector<contour> cntrs;
	for (int i = 0; i < contours.size(); i++)
	{
		if (!passed_vertex.at(i))
		{
			cntrs.push_back(contours.at(i));
		}
	}

	return cntrs;
}

void findContour(unsigned char** &img, contour &contour, int x, int y)
{
	vector<int> stack;
	vector<Point> cntr;

	int loop[2][10] = { { -1, -1, 0, 1, 1, 1, 0, -1, -1, -1 }, { 0, -1, -1, -1, 0, 1, 1, 1, 0, -1 } };
	int i = x, j = y;
	int step = 0;
	
	do
	{
		img[i][j] = 128;
		cntr.push_back(Point(i, j));

		int drct = 0;
		for (int k = 1; k < 9; k++)
		{
			if (img[i + loop[0][k]][j + loop[1][k]] == 255 &&
				(img[i + loop[0][k - 1]][j + loop[1][k - 1]] == 0 || img[i + loop[0][k + 1]][j + loop[1][k + 1]] == 0))
			{
				if (drct == 0)
				{
					drct = k;
				}
				else if (stack.size() == 0 || stack.back() != cntr.size())
				{
					stack.push_back(cntr.size());
				}
				else
				{
					break;
				}
			}
		}

		if (drct > 0)
		{
			i += loop[0][drct];
			j += loop[1][drct];
		}
		else if (stack.size() > 0 && (abs(x - i) + abs(y - j) > 2))
		{
			cntr.resize(stack.back());
			i = cntr.back().x;
			j = cntr.back().y;
			stack.pop_back();
		}
		else
		{
			break;
		}
		
	} while (!(i == x && j == y));
	
	i = 0;
	while (i < cntr.size())
	{
		contour.push_back(cntr.at(i));
		i += 1;
	}
}

void findContours_(unsigned char** &img, vector<contour> &contours)
{
	bool isBorder;
	//img with borders
	for (int i = 0; i < IMAGE_WIDTH; i++)
	{
		isBorder = false;
		for (int j = 0; j < IMAGE_HEIGTH; j++)
		{
			if (!isBorder && img[i][j] == 255)
			{
				contour cntr;
				findContour(img, cntr, i, j);

				if (cntr.size() > 0)
				{
					contours.push_back(cntr);
				}

				isBorder = true;
			}
			else if (img[i][j] == 0 && isBorder)
			{
				isBorder = false;
			}
			else if (img[i][j] == 128 && !isBorder)
			{
				isBorder = true;
			}
		}
	}
}

int pointDist(Point p1, Point p2)
{
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}


int mod(int x, int y)
{
	if (x < 0)
	{
		return y - 1;
	}
	if (x == y)
	{
		return 0;
	}

	return x;
}

bool thinning(contour &contour, int &start, int &end, vector<Point> &points)
{
	bool isFound;
	start = 0;
	end = 0;
	int i = 0, left, right, min_dist_diff = MAX_LINE_WIDTH * MAX_LINE_WIDTH;
	vector<Point> points_;

	while (i < contour.size() - 1)
	{
		left = mod(i - 1, contour.size());
		right = mod(i + 1, contour.size());

		int dist, dist_diff = 0;
		isFound = false;
		points_.clear();

		do
		{
			int dist_new;
			dist = pointDist(contour.at(left), contour.at(right));
			Point p;
			
			do
			{
				p.x = (contour.at(left).x + contour.at(right).x) / 2;
				p.y = (contour.at(left).y + contour.at(right).y) / 2;
				left = mod(left - 1, contour.size());
				dist_new = pointDist(contour.at(left), contour.at(right));
			} while (dist_new <= dist && right != left);

			right = mod(right + 1, contour.size());
			points_.push_back(p);

			if (dist > dist_diff)
			{
				dist_diff = dist;
			}
			
			isFound = abs(left - right) <= 1;
		} while (dist < MAX_LINE_WIDTH * MAX_LINE_WIDTH && !isFound);

		if (isFound && dist_diff < min_dist_diff)
		{
			start = i;
			end = left;
			min_dist_diff = dist_diff;
			points = points_;
		}
		
		i++;
	}

	return start != end;
}

vector<Point> approximation(vector<Point> &points, int x, int y)
{
	int max_dist = 0, max_i = 0;

	vector<Point> res, left, right;

	Line l = Line(points.at(x), points.at(y));

	for (int i = x + 1; i < y; i++)
	{
		int dist = l.distToLine(points.at(i));
		if (dist > max_dist)
		{
			max_dist = dist;
			max_i = i;
		}
	}

	if (max_dist > EPS_APROX)
	{
		left = approximation(points, x, max_i);
		right = approximation(points, max_i, y);

		left.pop_back();

		res.reserve(left.size() + right.size());
		res.insert(res.end(), left.begin(), left.end());
		res.insert(res.end(), right.begin(), right.end());
	}
	else
	{
		res.push_back(points.at(x));
		res.push_back(points.at(y));	
	}

	return res;
}

int main()
{
	VideoCapture capture("input.avi");

	Mat frame;
	vector<Vec4i> hierarchy;

	capture >> frame;

	int rows = frame.rows;
	int cols = frame.cols;

	VideoWriter writer("output.avi",
		capture.get(CV_CAP_PROP_FOURCC),
		capture.get(CV_CAP_PROP_FPS),
		Size(capture.get(CV_CAP_PROP_FRAME_WIDTH),
			capture.get(CV_CAP_PROP_FRAME_HEIGHT)));

	vector<Point2f> pts_src;
	pts_src.push_back(Point2f(3, 218));
	pts_src.push_back(Point2f(87, 150));
	pts_src.push_back(Point2f(263, 155));
	pts_src.push_back(Point2f(290, 227));

	vector<Point2f> pts_dst;
	pts_dst.push_back(Point2f(300, 608));
	pts_dst.push_back(Point2f(300, 500));
	pts_dst.push_back(Point2f(408, 500));
	pts_dst.push_back(Point2f(408, 608));

	Mat h = findHomography(pts_src, pts_dst);
	Mat h_inv = findHomography(pts_dst, pts_src);

	vector<Point2f> p1(1), p2(1);
	Mat roi_mask(IMAGE_WIDTH, IMAGE_HEIGTH, CV_8UC1, Scalar(0));
	
	//input is always (320, 240)
	for (int i = 0; i < IMAGE_HEIGTH; i++)
	{
		for (int j = 0; j < IMAGE_WIDTH; j++)
		{
			p1[0] = Point2f(i, j);
			perspectiveTransform(p1, p2, h_inv);
			if (p2[0].x >= 5 && p2[0].x <= 315 && p2[0].y >= 1 && p2[0].y <= 235)
			{
				roi_mask.at<uchar>(j, i) = 255;
			}
		}
	}
	
	float times[5] = { 0, 0, 0, 0, 0 };
	int t = 0;
	while(true)
	{
		if (!capture.read(frame))
		{
			break;
		}
		clock_t begin_time = clock();
		vector<vector<Point>> contours;

		Mat fr;
		frame.copyTo(fr);
		Mat black_color_mask;
		warpPerspective(frame, frame, h, Size(IMAGE_HEIGTH, IMAGE_WIDTH));
		GaussianBlur(frame, frame, Size(7, 7), 0);

		times[0] += float(clock() - begin_time) / CLOCKS_PER_SEC;

		begin_time = clock();
		
		cvtColor(frame, frame, CV_BGR2HSV);
		inRange(frame, Scalar(0, 0, 0, 0), Scalar(180, 255, 70, 0), black_color_mask);
		
		bitwise_and(black_color_mask, roi_mask, black_color_mask);

		times[1] += float(clock() - begin_time) / CLOCKS_PER_SEC;
		
		unsigned char **arr = new unsigned char*[black_color_mask.rows];
		for (int i = 0; i< black_color_mask.rows; ++i)
			arr[i] = new unsigned char[black_color_mask.cols];

		for (int i = 0; i < black_color_mask.rows; ++i)
			for (int j = 0; j < black_color_mask.cols; ++j) {
				arr[i][j] = (unsigned char)black_color_mask.at<uchar>(j, i);
			}
		for (int i = 0; i < black_color_mask.rows; ++i)
		{
			arr[i][0] = 0;
			arr[i][black_color_mask.cols - 1] = 0;
		}
		for (int j = 0; j < black_color_mask.cols; ++j)
		{
			arr[0][j] = 0;
			arr[black_color_mask.rows - 1][j] = 0;
		}
		begin_time = clock();
		findContours_(arr, contours);
		
		for (int i = 0; i < black_color_mask.rows; ++i)
			for (int j = 0; j < black_color_mask.cols; ++j) {
				black_color_mask.at<uchar>(j, i) = (uchar)arr[i][j];
			}

		times[2] += float(clock() - begin_time) / CLOCKS_PER_SEC;
		
		begin_time = clock();
		Graph graph(contours);
		times[3] += float(clock() - begin_time) / CLOCKS_PER_SEC;

		begin_time = clock();
		vector<contour> cntrs = filter_contours(contours, graph);
		times[4] += float(clock() - begin_time) / CLOCKS_PER_SEC;
		
		for (int i = 0; i < cntrs.size(); i++)
		{
			int x, y;
			vector<Point> points;

			if (thinning(cntrs.at(i), x, y, points); && points.size() > 1)
			{
				vector<Point> pnts = approximation(points, 0, points.size() - 1);
				vector<Point2f> pnts_(pnts.size());
				for (int j = 0; j < pnts.size(); j++)
				{
					Point p = pnts.at(j);
					pnts_.at(j) = Point2f(p.x, p.y);
				}
				perspectiveTransform(pnts_, pnts_, h_inv);
				for (int j = 0; j < pnts.size(); j++)
				{
					Point p = pnts_.at(j);
					pnts.at(j) = (Point((int)p.x, (int)p.y));
				}
				for (int k = 0; k + 1 < pnts.size(); k++)
				{
					line(fr, pnts.at(k), pnts.at(k + 1), Scalar(0, 0, 255));
				}
			}
		}
		
	
		for (int i = 0; i < black_color_mask.rows; ++i) {
			delete[] arr[i];
		}
		delete[] arr;
		
		
		writer.write(fr);
	}
	for (int i = 0; i < 5; i++)
	{
		std::cout << times[i] << '\n';
	}
	capture.release();
	writer.release();
	
	return 0;
}