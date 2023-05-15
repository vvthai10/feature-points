#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

#define PI 3.14
#define NUM_BINS 36
#define NUM_HIST 4
#define NUM_ORI 8

// Các struct sử dụng trong bài
// Lưu trữ thông tin các điểm đặc trưng
struct Keypoint {
	// Tọa độ rời rạc
	int i;
	int j;
	int octave;
	int scale;

	// Tọa độ liên tục
	float x;
	float y;
	float sigma;
	float extremum;

	vector<float> descriptor;
};
// Lưu trữ giá trị của đạo hàm theo x và y
struct Gradient {
	vector<vector<float>> Gx;
	vector<vector<float>> Gy;
};

// NOTE: Các struct dưới đây chỉ dùng cho thuật toán DoG
// Lưu trữ thông tin cập nhập trong thuật toán DoG
struct Offset {
	float s;
	float x;
	float y;
};
// Lưu trữ thông tin khi tính toán trong thuật toán DoG
struct ScaleSpacePyramid {
	int numOctaves;
	int imgPerOctave;
	vector<vector<Mat>> octaves;
	vector<vector<Gradient>> gradients;
};

// Các hàm được sử dụng chung
// Hàm chuyển từ kiểu Mat sang vector 2D
// - img: ảnh cần chuyển - chỉ xử lý ảnh xảm
// - type: 0 - chuyển theo kiểu uchar; 1 - chuyển theo kiểu float
vector<vector<float>> ConvertMat2Vector(const Mat& img, int type) {
	int rows = img.rows, cols = img.cols;
	vector<vector<float>> v_img;
	v_img.resize(rows, vector<float>(cols, 0));

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (type == 0) {
				v_img[i][j] = img.at <uchar>(i, j);
			}
			else {
				v_img[i][j] = img.at <float>(i, j);
			}
		}
	}

	return v_img;
}

// Hàm chuyển từ kiểu vector 2D sang Mat
// - img: vector cần chuyển - chỉ xử lý chuyển sang ảnh xám
// - type: 0 - chuyển theo kiểu uchar; 1 - chuyển theo kiểu float
Mat ConvertVector2Mat(const vector<vector<float>>& img, int type) {
	int rows = img.size(), cols = img[0].size();

	Size img_size(cols, rows);
	if (type == 0) {
		Mat m_img(img_size, CV_8U);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				m_img.at<uchar>(i, j) = img[i][j];
			}
		}
		return m_img;
	}
	else {
		Mat m_img(img_size, CV_32F);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				m_img.at<float>(i, j) = img[i][j];
			}
		}
		return m_img;
	}
}

// Hàm thêm padding vào ảnh
// - img: ảnh cần thêm padding
// - pad: số pixel cần thêm vào mỗi hướng
// - type: 0 - zero padding; 1 - replicate padding 
vector<vector<float>> ApplyPaddingImage(vector<vector<float>> img, int pad, int type) {
	int rows = img.size(), cols = img[0].size();
	vector<vector<float>> output;
	int padded_rows = rows + 2 * pad;
	int padded_cols = cols + 2 * pad;
	output.resize(padded_rows, vector<float>(padded_cols, 0));

	// zero padding
	if (type == 0) {
		for (int y = pad; y < rows + pad; y++) {
			for (int x = pad; x < cols + pad; x++) {
				output[y][x] = img[y - pad][x - pad];
			}
		}
	}
	// replicate padding
	else {
		for (int y = pad; y < rows + pad; y++) {
			for (int x = pad; x < cols + pad; x++) {
				output[y][x] = img[y - pad][x - pad];
			}
		}

		for (int y = 0; y < pad; y++) {
			// Pad trên và dưới
			for (int x = 0; x < padded_cols; x++) {
				output[y][x] = output[pad][x];
				output[padded_rows - 1 - y][x] = output[padded_rows - pad - 1][x];
			}

			// Pad trái và phải
			for (int x = 0; x < padded_rows; x++) {
				output[x][y] = output[x][pad];
				output[x][padded_cols - 1 - y] = output[x][padded_cols - pad - 1];
			}
		}
	}
	return output;
}

// Hàm in ảnh ra màn hình
// - img: ảnh cần hiển thị
// - title: Tên ảnh hiển thị
void ShowImage(const Mat& img, string title) {
	namedWindow(title, WINDOW_NORMAL);
	resizeWindow(title, img.cols / 2, img.rows / 2);
	imshow(title, img);
}

// Hàm tính tích chập
// - img: ảnh cần tích chập
// - kernel: ma trận dùng để tích chập với ảnh
vector<vector<float>> Convolve(const vector<vector<float>>& img, const vector<vector<float>>& kernel) {
	int rows = img.size(), cols = img[0].size();

	int size = kernel.size();
	int radius = size / 2;

	// Padding cho ảnh trước khi tích chập
	vector<vector<float>> pad = ApplyPaddingImage(img, radius, 1);

	// Tính tích chập ảnh và kernel
	vector<vector<float>> filtered;
	filtered.resize(rows, vector<float>(cols, 0));

	// Lẩy rows, cols theo ảnh mới
	rows = pad.size(), cols = pad[0].size();

	for (int y = radius; y < rows - radius; y++) {
		for (int x = radius; x < cols - radius; x++) {
			float sum = 0;
			for (int i = -radius; i <= radius; i++) {
				for (int j = -radius; j <= radius; j++) {
					sum += pad[y + i][x + j] * kernel[i + radius][j + radius];
				}
			}
			filtered[y - radius][x - radius] = sum;
		}
	}

	return filtered;
}



// THUẬT TOÁN 1: Các hàm sử dụng để xử lý thao tác với thuật toán Harris
// ---------------------------------------------------------------------
// Hàm hỗ trợ để sắp xếp giảm dần
bool compareSecond(const pair<Point2f, float>& a, const pair<Point2f, float>& b) {
	return a.second > b.second;
}

// Hàm tính các ma trận đạo hàm theo x và y dựa vào gaussian
// - gaussian: ảnh đã được gaussian
// - x_grad: ma trận đạo hàm theo x
// - y_grad: ma trận đạo hàm theo y
void CalcGradientByGaussian(const vector<vector<float>>& gaussian, vector<vector<float>>& x_grad, vector<vector<float>>& y_grad) {
	int rows = gaussian.size(), cols = gaussian[0].size();
	x_grad.resize(rows, vector<float>(cols, 0));
	y_grad.resize(rows, vector<float>(cols, 0));

	float gx, gy;
	for (int y = 1; y < rows - 1; y++) {
		for (int x = 1; x < cols - 1; x++) {
			gx = (gaussian[y][x + 1] - gaussian[y][x - 1]) * 0.5;
			x_grad[y][x] = gx;
			gy = (gaussian[y + 1][x] - gaussian[y - 1][x]) * 0.5;
			y_grad[y][x] = gy;
		}
	}
}

// Hàm tạo ra kernel gaussian
// - sigma: giá trị sigma dùng để tính kernel
vector<vector<float>> CreateGaussianKernel(float sigma) {
	int size = ceil(6 * sigma);
	size = (size % 2 == 0) ? size + 1 : size;
	int radius = size / 2;

	// Tạo ma trận kernel
	vector<vector<float>> kernel;
	kernel.resize(size, vector<float>(size, 0));

	for (int y = -radius; y <= radius; y++) {
		for (int x = -radius; x <= radius; x++) {
			float val = exp(-(x * x + y * y) / (2 * sigma * sigma));
			val = val / (2 * PI * sigma * sigma);
			kernel[y + radius][x + radius] = val;
		}
	}

	return kernel;
}

// Hàm lấy đạo hàm theo x và y - dùng cho SIFT
Gradient GetGradientOfHarris(const Mat& img) {
	int rows = img.rows, cols = img.cols;

	// Chuyển ảnh sang gray-scale
	Mat gray;
	cvtColor(img, gray, COLOR_RGB2GRAY);

	// Chuyển ảnh sang vector 2D để xử lý
	// Đoạn dưới này sẽ xử lý bằng ảnh GRAY
	vector<vector<float>> v_img = ConvertMat2Vector(gray, 0);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			v_img[i][j] /= 255.0;
		}
	}

	// Apply Gaussian Blur with sigma = 1
	// - Tính kernel 
	vector<vector<float>> kernel = CreateGaussianKernel(1);
	//// - Tính tích chập giữa ảnh và kernel
	vector<vector<float>> filtered = Convolve(v_img, kernel);

	// Tìm ma trận đạo hàm theo x, y bằng gaussian
	vector<vector<float>> x_grad, y_grad;
	CalcGradientByGaussian(filtered, x_grad, y_grad);

	Gradient res;
	res.Gx = x_grad;
	res.Gy = y_grad;
	return res;
}

// Hàm tổng hợp các bước xử lý của thuật toán Harris
vector<Keypoint> HandleDetectByHarris(const Mat& img) {
	int rows = img.rows, cols = img.cols;

	// Chuyển ảnh sang gray-scale
	Mat gray;
	cvtColor(img, gray, COLOR_RGB2GRAY);

	// Chuyển ảnh sang vector 2D để xử lý
	// Đoạn dưới này sẽ xử lý bằng ảnh GRAY
	vector<vector<float>> v_img = ConvertMat2Vector(gray, 0);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			v_img[i][j] /= 255.0;
		}
	}

	// Tích chập với gaussian filter với sigma = 1
	// - Tính kernel 
	vector<vector<float>> kernel = CreateGaussianKernel(1);
	//// - Tính tích chập giữa ảnh và kernel
	vector<vector<float>> filtered = Convolve(v_img, kernel);

	// Tìm ma trận đạo hàm theo x, y bằng sobel filter
	vector<vector<float>> x_grad, y_grad;
	CalcGradientByGaussian(filtered, x_grad, y_grad);
	//CalcGradientBySobel(filtered, x_grad, y_grad);

	// Tính xx, yy, xy 
	vector<vector<float>> xx_grad, yy_grad, xy_grad;
	xx_grad.resize(rows, vector<float>(cols, 0));
	yy_grad.resize(rows, vector<float>(cols, 0));
	xy_grad.resize(rows, vector<float>(cols, 0));
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			xx_grad[y][x] = pow(x_grad[y][x], 2);
			yy_grad[y][x] = pow(y_grad[y][x], 2);
			xy_grad[y][x] = x_grad[y][x] * y_grad[y][x];
		}
	}

	float k = 0.04;
	int window_size = 3;
	float ratio = 0.001;

	// Tính các giá trị để tìm các diểm đặc trưng
	vector<vector<float>> r_harris;
	r_harris.resize(rows, vector<float>(cols, 0));
	int offset = window_size / 2;
	float start_y, end_y, start_x, end_x;
	float det, trace, r;
	for (int y = offset; y < rows - offset; y++) {
		for (int x = offset; x < cols - offset; x++) {
			// Kích thước của window
			start_y = y - offset;
			end_y = y + offset + 1;
			start_x = x - offset;
			end_x = x + offset + 1;

			float s_xx = 0, s_yy = 0, s_xy = 0;
			for (int i = start_y; i < end_y; i++) {
				for (int j = start_x; j < end_x; j++) {
					s_xx += xx_grad[i][j];
					s_yy += yy_grad[i][j];
					s_xy += xy_grad[i][j];
				}
			}
			det = (s_xx * s_yy) - (s_xy * s_xy);
			trace = s_xx + s_yy;

			r = det - k * (trace * trace);
			r_harris[y][x] = r;
		}
	}

	// Tìm các điểm đặc trưng thỏa điều kiện
	float max_val = -999;
	for (int y = offset; y < rows - offset; y++) {
		for (int x = offset; x < cols - offset; x++) {
			max_val = (max_val > r_harris[y][x]) ? max_val : r_harris[y][x];
		}
	}

	vector<pair<Point2f, float>> corners;
	for (int y = offset; y < rows - offset; y++) {
		for (int x = offset; x < cols - offset; x++) {
			if (r_harris[y][x] > ratio * max_val) {
				corners.push_back({ Point(x, y) , r_harris[y][x] });
			}
		}
	}

	// Non maximal suppression
	sort(corners.begin(), corners.end(), compareSecond);
	vector<Keypoint> kps;
	Keypoint kp;
	kp.x = corners[0].first.x, kp.y = corners[0].first.y;
	kps.push_back(kp);
	float dis = 10;
	for (int i = 0; i < corners.size(); i++) {
		bool check = true;
		for (int j = 0; j < kps.size(); j++) {
			if (abs(kps[j].x - corners[i].first.x) <= dis && abs(kps[j].y - corners[i].first.y) <= dis) {
				check = false;
				break;
			}
		}

		if (check) {
			kp.x = corners[i].first.x;
			kp.y = corners[i].first.y;
			kps.push_back(kp);
		}
	}

	return kps;
}

// Hàm xử lý và vẽ các điểm đặc trưng lên ảnh
Mat detectHarris(const Mat& img) {
	// Tìm các điểm keypoint
	vector<Keypoint> kps = HandleDetectByHarris(img);
	cout << "The number of feature points: " << kps.size() << endl;
	// Vẽ các điểm đặc trưng
	for (int i = 0; i < kps.size(); i++) {
		circle(img, Point(kps[i].x, kps[i].y), 5, Scalar(0, 255, 0), 1, LINE_AA);
	}
	return img;
}



// THUẬT TOÁN 2: Các hàm sử dụng để xử lý thao tác với thuật toán Blob
// -------------------------------------------------------------------
// Hàm lấy thông tin của các đạo hàm theo x và y - dùng cho SIFT
vector<Gradient> GetGradientOfBlob(const Mat& img) {
	int rows = img.rows, cols = img.cols;

	// Khởi tạo các thông số mặc định
	int n_scales = 5;
	float k = 1.24;
	float df_sigma = sqrt(2);
	float threshold = 0.02;

	// Tính list sigma
	vector<float> list_sigma;
	for (int i = 0; i < n_scales; i++) {
		float val;
		val = df_sigma * pow(k, i);
		list_sigma.push_back(val);
	}

	// Tính các ma trận LoG
	vector<vector<vector<float>>> list_Gauss;
	for (int i = 0; i < n_scales; i++) {
		// Tính kích thước của ma trận LoG
		int size_kernel = ceil(6 * list_sigma[i]);
		size_kernel = (size_kernel % 2 == 0) ? size_kernel + 1 : size_kernel;

		vector<vector<float>> kernel;
		kernel.resize(size_kernel, vector<float>(size_kernel, 0));
		int radius = size_kernel / 2;
		for (int y = -radius; y <= radius; y++) {
			for (int x = -radius; x <= radius; x++) {
				float val = exp(-(x * x + y * y) / (2 * list_sigma[i] * list_sigma[i]));
				val = val / (2 * PI * list_sigma[i] * list_sigma[i]);
				kernel[y + radius][x + radius] = val;
			}
		}

		list_Gauss.push_back(kernel);
	}

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	vector<vector<float>> v_img = ConvertMat2Vector(gray, 0);
	rows = v_img.size();
	cols = v_img[0].size();
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			v_img[i][j] /= 255.0;
		}
	}

	// Tính tích chập giữa ảnh với đúng thứ tự scale
	vector<vector<vector<float>>> list_filtered;
	for (int i = 0; i < n_scales; i++) {
		vector<vector<float>> filtered = Convolve(v_img, list_Gauss[i]);
		list_filtered.push_back(filtered);
	}

	// Tính các gradient theo các sigma khác khau
	vector<Gradient> list_gradient;
	for (int i = 0; i < n_scales; i++) {
		float gx, gy;
		const vector<vector<float>>& gaussian = list_filtered[i];
		Gradient grad;
		grad.Gx.resize(rows, vector<float>(cols, 0));
		grad.Gy.resize(rows, vector<float>(cols, 0));
		for (int y = 1; y < rows - 1; y++) {
			for (int x = 1; x < cols - 1; x++) {
				gx = (gaussian[y][x + 1] - gaussian[y][x - 1]) * 0.5;
				grad.Gx[y][x] = gx;
				gy = (gaussian[y + 1][x] - gaussian[y - 1][x]) * 0.5;
				grad.Gy[y][x] = gy;
			}
		}
		list_gradient.push_back(grad);
	}

	return list_gradient;
}

// Hàm tổng hợp các bước xử lý của thuật toán Blob
vector<Keypoint> HandleDetectByBlob(const Mat& img) {
	int rows = img.rows, cols = img.cols;

	// Khởi tạo các thông số mặc định
	int n_scales = 5;
	float k = 1.24;
	float df_sigma = sqrt(2);
	float threshold = 0.02;

	// Tính list sigma
	vector<float> list_sigma;
	for (int i = 0; i < n_scales; i++) {
		float val;
		val = df_sigma * pow(k, i);
		list_sigma.push_back(val);
	}

	// Tính các ma trận LoG
	vector<vector<vector<float>>> list_LoG;
	for (int i = 0; i < n_scales; i++) {
		// Tính kích thước của ma trận LoG
		int size_kernel = ceil(6 * list_sigma[i]);
		size_kernel = (size_kernel % 2 == 0) ? size_kernel + 1 : size_kernel;

		vector<vector<float>> LoG;
		LoG.resize(size_kernel, vector<float>(size_kernel, 0));
		int radius = size_kernel / 2;
		for (int y = -radius; y <= radius; y++) {
			for (int x = -radius; x <= radius; x++) {
				float val = exp(-(x * x + y * y) / (2 * list_sigma[i] * list_sigma[i]));
				val = val * (1 - (x * x + y * y) / (2 * list_sigma[i] * list_sigma[i]));
				val = val * (-1 / (PI * pow(list_sigma[i], 2)));
				LoG[y + radius][x + radius] = val;
			}
		}

		list_LoG.push_back(LoG);
	}

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	vector<vector<float>> v_img = ConvertMat2Vector(gray, 0);
	rows = v_img.size();
	cols = v_img[0].size();
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			v_img[i][j] /= 255.0;
		}
	}

	// Tính tích chập giữa ảnh với đúng thứ tự scale
	vector<vector<vector<float>>> list_laplacian;
	for (int i = 0; i < n_scales; i++) {
		vector<vector<float>> laplacian = Convolve(v_img, list_LoG[i]);
		list_laplacian.push_back(laplacian);
	}

	// Tính các điểm đặc trưng
	vector<Keypoint> kps;
	for (int n = 0; n < n_scales; n++) {
		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				bool check = true;
				for (int s = -1; s <= 1; s++) {
					for (int i = -1; i <= 1; i++) {
						for (int j = -1; j <= 1; j++) {
							if (y + i >= 0 && x + j >= 0 && y + i < rows && x + j < cols && n + s >= 0 && n + s < n_scales) {
								if (list_laplacian[n][y][x] > threshold) {
									if (list_laplacian[n][y][x] < list_laplacian[n + s][y + i][x + j]) {
										check = false;
									}
								}
								else {
									check = false;
								}
							}
						}
					}
				}
				if (check) {
					Keypoint blob;
					blob.x = x, blob.y = y, blob.scale = n, blob.sigma = list_sigma[n];
					kps.push_back(blob);

				}
			}
		}
	}

	return kps;
}

// Hàm tìm các keypoint và vẽ các điểm đặc trưng vào ảnh
Mat detectBlob(const Mat& img) {
	vector<Keypoint> kps = HandleDetectByBlob(img);
	cout << "The number of feature points: " << kps.size() << endl;
	for (int i = 0; i < kps.size(); i++)
	{
		float radius = int(kps[i].sigma * sqrt(2));
		circle(img, Point(kps[i].x, kps[i].y), radius, Scalar(20, 255, 20), 1, LINE_AA);
	}

	return img;
}



// THUẬT TOÁN 3: Các hàm sử dụng để xử lý thao tác với thuật toán DoG
// ------------------------------------------------------------------
// Hàm lấy các giá trị pixel của ảnh
float GetValue(const Mat& img, int x, int y) {
	int rows = img.rows, cols = img.cols;
	if (x < 0) {
		x = 0;
	}
	if (x >= cols) {
		x = cols - 1;
	}
	if (y < 0) {
		y = 0;
	}
	if (y >= rows) {
		y = rows - 1;
	}

	return img.at<float>(y, x);
}

// Hàm kiểm tra điểm đang xét với 26 điểm xung quanh
bool PointIsExtremum(const vector<Mat>& octave, int scale, int x, int y)
{
	const Mat& img = octave[scale];
	const Mat& prev = octave[scale - 1];
	const Mat& next = octave[scale + 1];

	bool is_min = true, is_max = true;
	float val = img.at<float>(y, x), neighbor;

	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			neighbor = prev.at<float>(y + i, x + j);
			if (neighbor > val) is_max = false;
			if (neighbor < val) is_min = false;

			neighbor = next.at<float>(y + i, x + j);
			if (neighbor > val) is_max = false;
			if (neighbor < val) is_min = false;

			neighbor = img.at<float>(y + i, x + j);
			if (neighbor > val) is_max = false;
			if (neighbor < val) is_min = false;

			if (!is_min && !is_max) return false;
		}
	}
	return true;
}

// Làm mờ ảnh bằng gaussian với giá trị sigma
Mat GaussianBlur(const Mat& img, float sigma) {
	int rows = img.rows, cols = img.cols;
	int size = ceil(6 * sigma);
	size = (size % 2 == 0) ? size + 1 : size;
	int radius = size / 2;

	// Tạo ma trận kernel
	vector<vector<float>> kernel;
	kernel.resize(size, vector<float>(size, 0));

	float sum = 0;
	for (int y = -radius; y <= radius; y++) {
		for (int x = -radius; x <= radius; x++) {
			float val = exp(-(x * x + y * y) / (2 * sigma * sigma));
			val = val / (2 * PI * sigma * sigma);
			sum += val;
			kernel[y + radius][x + radius] = val;
		}
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			kernel[i][j] /= sum;
		}
	}

	// Chuyển Mat thành vector để tích chập
	vector<vector<float>> v_img = ConvertMat2Vector(img, 1);
	vector<vector<float>> filtered = Convolve(v_img, kernel);


	return ConvertVector2Mat(filtered, 1);
}

// Tính "pyramid" gaussian 
ScaleSpacePyramid ComputeGaussianPyramid(int n_oct, int imgs_per, const vector<float> list_sigma, const Mat& img) {
	ScaleSpacePyramid gaussian_pyramid;
	gaussian_pyramid.numOctaves = n_oct;
	gaussian_pyramid.imgPerOctave = imgs_per;
	gaussian_pyramid.octaves.resize(n_oct);
	Mat base_img = img;
	for (int i = 0; i < n_oct; i++) {
		gaussian_pyramid.octaves[i].reserve(gaussian_pyramid.imgPerOctave);
		gaussian_pyramid.octaves[i].push_back(base_img);
		// Chỉ tính thêm n - 1 ảnh trong từng octave.
		for (int j = 1; j < list_sigma.size(); j++) {
			const Mat& prev = gaussian_pyramid.octaves[i].back();
			Mat next = GaussianBlur(prev, list_sigma[j]);
			gaussian_pyramid.octaves[i].push_back(next);
		}
		// Thêm ảnh thứ 0 vào octave thứ i + 1
		const Mat& next = gaussian_pyramid.octaves[i][imgs_per - 3];
		// Resize ảnh bằng N * delta_min / delta_in
		Size next_size(next.cols / 2, next.rows / 2);
		resize(next, base_img, next_size, cv::INTER_NEAREST);

	}

	return gaussian_pyramid;
}

// Tính "pyramid" DoG 
ScaleSpacePyramid ComputeDoGPyramid(const ScaleSpacePyramid& gaussian_pyramid) {
	ScaleSpacePyramid dog_pyramid;
	dog_pyramid.numOctaves = gaussian_pyramid.numOctaves;
	dog_pyramid.imgPerOctave = gaussian_pyramid.imgPerOctave - 1;
	dog_pyramid.octaves.resize(gaussian_pyramid.numOctaves);

	for (int i = 0; i < dog_pyramid.numOctaves; i++) {
		dog_pyramid.octaves[i].reserve(dog_pyramid.imgPerOctave);
		for (int j = 1; j < gaussian_pyramid.imgPerOctave; j++) {
			Mat sub;
			subtract(gaussian_pyramid.octaves[i][j], gaussian_pyramid.octaves[i][j - 1], sub);
			dog_pyramid.octaves[i].push_back(sub);
		}
	}
	return dog_pyramid;
}

// Tính các giá trị để cập nhật cho tạo độ x, y, sigma
Offset QuadraticInterpolation(Keypoint& kp, const vector<Mat>& octave) {
	int scale = kp.scale;
	const Mat& img = octave[scale];
	const Mat& prev = octave[scale - 1];
	const Mat& next = octave[scale + 1];

	float g1, g2, g3;
	float h11, h12, h13, h22, h23, h33;
	int x = kp.i, y = kp.j;

	// Tính toán độ dốc 
	g1 = (GetValue(next, x, y) - GetValue(prev, x, y)) * 0.5;
	g2 = (GetValue(img, x + 1, y) - GetValue(img, x - 1, y)) * 0.5;
	g3 = (GetValue(img, x, y + 1) - GetValue(img, x, y - 1)) * 0.5;

	// Ma trận hessian 3x3
	h11 = GetValue(next, x, y) + GetValue(prev, x, y) - 2 * GetValue(img, x, y);
	h22 = GetValue(img, x + 1, y) + GetValue(img, x - 1, y) - 2 * GetValue(img, x, y);
	h33 = GetValue(img, x, y + 1) + GetValue(img, x, y - 1) - 2 * GetValue(img, x, y);
	h12 = (GetValue(next, x + 1, y) - GetValue(next, x - 1, y) - GetValue(prev, x + 1, y) + GetValue(prev, x - 1, y)) * 0.25;
	h13 = (GetValue(next, x, y + 1) - GetValue(next, x, y - 1) - GetValue(prev, x, y + 1) + GetValue(prev, x, y - 1)) * 0.25;
	h23 = (GetValue(img, x + 1, y + 1) - GetValue(img, x + 1, y - 1) - GetValue(img, x - 1, y + 1) + GetValue(img, x - 1, y - 1)) * 0.25;

	// Ma trận nghịch đảo hessian 3x3
	float inv_h11, inv_h12, inv_h13, inv_h22, inv_h23, inv_h33;
	float det = h11 * h22 * h33 - h11 * h23 * h23 - h12 * h12 * h33 + 2 * h12 * h13 * h23 - h13 * h13 * h22;
	inv_h11 = (h22 * h33 - h23 * h23) / det;
	inv_h12 = (h13 * h23 - h12 * h33) / det;
	inv_h13 = (h12 * h23 - h13 * h22) / det;
	inv_h22 = (h11 * h33 - h13 * h13) / det;
	inv_h23 = (h12 * h13 - h11 * h23) / det;
	inv_h33 = (h11 * h22 - h12 * h12) / det;

	// Tính cực trị liên tục...
	float offset_s = -inv_h11 * g1 - inv_h12 * g2 - inv_h13 * g3;
	float offset_x = -inv_h12 * g1 - inv_h22 * g2 - inv_h23 * g3;
	float offset_y = -inv_h13 * g1 - inv_h23 * g3 - inv_h33 * g3;

	// Giá trị dùng để loại bỏ các keypoint không tương phản
	float interpolated_extrema_val = GetValue(img, x, y) + 0.5 * (g1 * offset_s + g2 * offset_x + g3 * offset_y);
	kp.extremum = interpolated_extrema_val;
	return Offset{ offset_s, offset_x, offset_y };
}

// Kiểm tra điểm (x, y) có nằm trên cạnh không
bool KeypointOnEdge(const Keypoint& point, const vector<Mat>& octave) {
	const Mat& img = octave[point.scale];
	float h11, h12, h22;
	int x = point.i, y = point.j;

	h11 = GetValue(img, x + 1, y) + GetValue(img, x - 1, y) - 2 * GetValue(img, x, y);
	h22 = GetValue(img, x, y + 1) + GetValue(img, x, y - 1) - 2 * GetValue(img, x, y);
	h12 = (GetValue(img, x + 1, y + 1) - GetValue(img, x + 1, y - 1) - GetValue(img, x - 1, y + 1) + GetValue(img, x - 1, y - 1)) * 0.25;

	float det = h11 * h22 - h12 * h12;
	float tr = h11 + h22;
	float edge = tr * tr / det;
	//  12.1
	if (edge > 12.1) {
		return true;
	}
	return false;
}

// Tổng hợp các bước xử lý để chọn ra các điểm đặc trưng
bool RefineOrDiscardKeypoint(Keypoint& kp, const vector<Mat>& octave, float scales_per_octave, float sigma_min, float delta_min) {
	// Điều kiện dừng:
	// - Khi max(a*) < 0.6 hoặc lặp 5 lần
	bool check = false;
	for (int i = 0; i < 5; i++) {
		Offset offset = QuadraticInterpolation(kp, octave);
		float max_offset = max({ abs(offset.s), abs(offset.x), abs(offset.y) });

		// Cập nhật vị trí nội suy
		kp.scale += round(offset.s);
		kp.i += round(offset.x);
		kp.j += round(offset.y);

		if (kp.scale >= octave.size() - 1 || kp.scale < 1) {
			break;
		}

		// Kiểm tra độ tương phản
		float c_dog = 0.015;
		bool contrasted = abs(kp.extremum) > c_dog;
		// Loại bỏ các keypoint trên cạnh
		bool onEdge = KeypointOnEdge(kp, octave);
		if (max_offset < 0.6 && contrasted && !onEdge) {
			// Cập nhật tọa độ tuyệt đối tương ứng
			kp.sigma = pow(2, kp.octave) * sigma_min * pow(2, (offset.s + kp.scale) / scales_per_octave);
			kp.x = delta_min * pow(2, kp.octave) * (offset.x + kp.i);
			kp.y = delta_min * pow(2, kp.octave) * (offset.y + kp.j);

			check = true;
			break;
		}
	}
	return check;
}

// Hàm lấy "pyramid" đạo hàm theo x và y - dùng cho SIFT
ScaleSpacePyramid GetGradientOfDoG(const Mat& img) {
	int rows = img.rows, cols = img.cols;

	// Các bước xử lý ảnh đầu vào
		// 1. Chuyển ảnh sang ảnh xám
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	// 2. Chuyển ảnh sang vector để tính toán
	vector<vector<float>> v_img = ConvertMat2Vector(gray, 0);

	// 3. Chuẩn hóa ảnh
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			v_img[i][j] /= 255;
		}
	}

	// Khởi tạo các tham số
	float delta_min = 0.5;
	float sigma_min = 0.8;
	float sigma = sigma_min / delta_min;
	float scales_per_octave = 3;
	float k = pow(2, 1 / scales_per_octave);
	//float k = sqrt(2);

	int n_octave = 8;
	int imgs_per_octave = scales_per_octave + 3;

	// Computation of the digital Gaussian scale-space
		// Interpolate the original image 
	Mat nor_img = ConvertVector2Mat(v_img, 1);
	Size new_size(cols / delta_min, rows / delta_min);
	Mat interpolated_img;
	resize(nor_img, interpolated_img, new_size, cv::INTER_LINEAR);

	// Blur the interpolated image
	// sigma = (sigma_min^2 - sigma_in^2 )/delta_min 
	Mat base_img = GaussianBlur(interpolated_img, sqrt(sigma * sigma - 1));

	// Tạo danh sách các giá trị sigma 
	vector<float> list_sigma;
	list_sigma.push_back(sigma);
	for (int i = 1; i < imgs_per_octave; i++) {
		float a = sigma * pow(k, i - 1);
		float b = sigma * pow(k, i);
		float val = sqrt(b * b - a * a);
		list_sigma.push_back(val);
	}

	// Tạo một biến quản lý các ma trận Gaussin blur
	ScaleSpacePyramid gaussian_pyramid = ComputeGaussianPyramid(n_octave, imgs_per_octave, list_sigma, base_img);

	ScaleSpacePyramid grad_pyramid;
	grad_pyramid.numOctaves = gaussian_pyramid.numOctaves;
	grad_pyramid.imgPerOctave = gaussian_pyramid.imgPerOctave;
	grad_pyramid.gradients.resize(gaussian_pyramid.numOctaves);
	float gx, gy;

	for (int i = 0; i < gaussian_pyramid.numOctaves; i++) {
		grad_pyramid.gradients[i].resize(gaussian_pyramid.imgPerOctave);
		int rows_cur = gaussian_pyramid.octaves[i][0].rows;
		int cols_cur = gaussian_pyramid.octaves[i][0].cols;
		for (int j = 0; j < gaussian_pyramid.imgPerOctave; j++) {
			grad_pyramid.gradients[i][j].Gx.resize(rows_cur, vector<float>(cols_cur, 0));
			grad_pyramid.gradients[i][j].Gy.resize(rows_cur, vector<float>(cols_cur, 0));
			for (int y = 1; y < rows_cur - 1; y++) {
				for (int x = 1; x < cols_cur - 1; x++) {
					gx = (GetValue(gaussian_pyramid.octaves[i][j], x + 1, y) - GetValue(gaussian_pyramid.octaves[i][j], x - 1, y)) * 0.5;
					grad_pyramid.gradients[i][j].Gx[y][x] = gx;
					gy = (GetValue(gaussian_pyramid.octaves[i][j], x, y + 1) - GetValue(gaussian_pyramid.octaves[i][j], x, y - 1)) * 0.5;
					grad_pyramid.gradients[i][j].Gy[y][x] = gy;
				}
			}
		}
	}

	return grad_pyramid;
}

// Tổng hợp các bước xử lý thuật toán DoG
vector<Keypoint> HandleDetectByDoG(const Mat& img) {
	int rows = img.rows, cols = img.cols;

	// Các bước xử lý ảnh đầu vào
		// 1. Chuyển ảnh sang ảnh xám
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	// 2. Chuyển ảnh sang vector để tính toán
	vector<vector<float>> v_img = ConvertMat2Vector(gray, 0);

	// 3. Chuẩn hóa ảnh
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			v_img[i][j] /= 255;
		}
	}

	// Khởi tạo các tham số
	float delta_min = 0.5;
	float sigma_min = 0.8;
	float sigma = sigma_min / delta_min;
	float scales_per_octave = 3;
	float k = pow(2, 1 / scales_per_octave);
	//float k = sqrt(2);

	int n_octave = 8;
	int imgs_per_octave = scales_per_octave + 3;

	// Computation of the digital Gaussian scale-space
		// Interpolate the original image 
	Mat nor_img = ConvertVector2Mat(v_img, 1);
	Size new_size(cols / delta_min, rows / delta_min);
	Mat interpolated_img;
	resize(nor_img, interpolated_img, new_size, cv::INTER_LINEAR);

	// Blur the interpolated image
	// sigma = (sigma_min^2 - sigma_in^2 )/delta_min 
	Mat base_img = GaussianBlur(interpolated_img, sqrt(sigma * sigma - 1));

	// Tạo danh sách các giá trị sigma 
	vector<float> list_sigma;
	list_sigma.push_back(sigma);
	for (int i = 1; i < imgs_per_octave; i++) {
		float a = sigma * pow(k, i - 1);
		float b = sigma * pow(k, i);
		float val = sqrt(b * b - a * a);
		list_sigma.push_back(val);
	}

	// Tạo một biến quản lý các ma trận Gaussin blur
	ScaleSpacePyramid gaussian_pyramid = ComputeGaussianPyramid(n_octave, imgs_per_octave, list_sigma, base_img);

	// Tạo một biến quản lý các DoG
	ScaleSpacePyramid dog_pyramid = ComputeDoGPyramid(gaussian_pyramid);

	// Tìm các điểm đặc trưng
	vector<Keypoint> kps;
	float threshold = 0;
	for (int i = 0; i < dog_pyramid.numOctaves; i++) {
		const vector<Mat>& octave = dog_pyramid.octaves[i];
		for (int j = 1; j < dog_pyramid.imgPerOctave - 1; j++) {
			const Mat& cur_img = octave[j];
			for (int y = 1; y < cur_img.rows - 1; y += 1) {
				for (int x = 1; x < cur_img.cols - 1; x += 1) {
					if (abs(cur_img.at<float>(y, x)) > threshold) {
						if (PointIsExtremum(octave, j, x, y)) {
							Keypoint kp = { x, y, i, j , -1, -1, -1, -1 };
							if (RefineOrDiscardKeypoint(kp, octave, scales_per_octave, sigma_min, delta_min)) {
								kps.push_back(kp);
							}
						}
					}
				}
			}
		}
	}
	return kps;
}

// Hàm tìm các keypoint và vẽ các điểm đặc trưng vào ảnh 
Mat detectDog(const Mat& img) {
	vector<Keypoint> kps = HandleDetectByDoG(img);
	cout << "The number of feature points: " << kps.size() << endl;
	
	for (int i = 0; i < kps.size(); i++)
	{
		float radius = int(kps[i].sigma * sqrt(2));
		circle(img, Point2f(kps[i].x, kps[i].y), radius, Scalar(0, 0, 255), 1, LINE_AA);
	}

	return img;
}



// THUẬT TOÁN 4: Các hàm sử dụng để xử lý thao tác với thuật toán SIFT
// -------------------------------------------------------------------
// Hàm lấy giá trị pixel trong ma trận
float GetValueMatrix(const vector<vector<float>>& img, int x, int y) {
	int rows = img.size(), cols = img[0].size();
	if (x < 0) {
		x = 0;
	}
	if (x >= cols) {
		x = cols - 1;
	}
	if (y < 0) {
		y = 0;
	}
	if (y >= rows) {
		y = rows - 1;
	}

	return img[y][x];
}

// Cập nhật thông số của histogram - dùng trong bước Discription
void UpdateHistograms(float hist[NUM_HIST][NUM_HIST][NUM_ORI], float x, float y, float total, float theta, float lambda_desc)
{
	float x_i, y_j;
	for (int i = 1; i <= NUM_HIST; i++) {
		x_i = (i - (1 + (float)NUM_HIST) / 2) * 2 * lambda_desc / NUM_HIST;
		if (abs(x_i - x) <= 2 * lambda_desc / NUM_HIST) {
			for (int j = 1; j <= NUM_HIST; j++) {
				y_j = (j - (1 + (float)NUM_HIST) / 2) * 2 * lambda_desc / NUM_HIST;
				if (abs(y_j - y) <= 2 * lambda_desc / NUM_HIST) {

					float histWeight = (1 - NUM_HIST * 0.5 / lambda_desc * abs(x_i - x)) * (1 - NUM_HIST * 0.5 / lambda_desc * abs(y_j - y));

					for (int k = 1; k <= NUM_ORI; k++) {
						float theta_k = 2 * PI * (k - 1) / NUM_ORI;
						float theta_diff = fmod(theta_k - theta + 2 * PI, 2 * PI);
						if (abs(theta_diff) < 2 * PI / NUM_ORI) {
							float binWeight = 1 - NUM_ORI * 0.5 / PI * abs(theta_diff);
							hist[i - 1][j - 1][k - 1] += histWeight * binWeight * total;
						}
					}
				}
			}
		}
	}
}

// Bước xử lý orientation và description theo thuật toán Harris
vector<float> HandleOrientationOfHarris(Keypoint& kp, const Gradient& grad) {
	float pix_dist = 1;
	float lambda_ori = 1.5;
	float lambda_desc = 6;
	kp.sigma = 1;
	vector<float> orientations;

	const vector<vector<float>>& gradX = grad.Gx;
	const vector<vector<float>>& gradY = grad.Gy;
	int rows = gradX.size(), cols = gradX[0].size();

	// Kiểm tra điểm kp có nằm gần biên ảnh không
	float min_dist_from_border = min({ kp.x, kp.y, pix_dist * cols - kp.x, pix_dist * rows - kp.y });
	if (min_dist_from_border <= sqrt(2) * lambda_desc * kp.sigma) {
		return orientations;
	}

	float hist[NUM_BINS] = {0};
	int bin;
	float gx, gy, grad_norm, weight, theta;
	float patch_sigma = lambda_ori * kp.sigma;
	float patch_radius = 3 * patch_sigma;
	int x_start = round((kp.x - patch_radius) / pix_dist);
	int x_end = round((kp.x + patch_radius) / pix_dist);
	int y_start = round((kp.y - patch_radius) / pix_dist);
	int y_end = round((kp.y + patch_radius) / pix_dist);

	// Tính toán orientation histogram
	for (int x = x_start; x <= x_end; x++) {
		for (int y = y_start; y <= y_end; y++) {
			gx = GetValueMatrix(gradX, x, y);
			gy = GetValueMatrix(gradY, x, y);
			grad_norm = sqrt(gx * gx + gy * gy);
			weight = exp(-(pow(x * pix_dist - kp.x, 2) + pow(y * pix_dist - kp.y, 2))
				/ (2 * patch_sigma * patch_sigma));
			theta = fmod(atan2(gy, gx) + 2 * PI, 2 * PI);
			bin = (int)round(NUM_BINS / (2 * PI) * theta) % NUM_BINS;
			hist[bin] += weight * grad_norm;
		}
	}

	// Thực hiện làm trơn hist, thực hiện 6 lần bằng filter [1, 1, 1] / 3
	float smoothHist[NUM_BINS];
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < NUM_BINS; j++) {
			int prev = (j - 1 + NUM_BINS) % NUM_BINS;
			int next = (j + 1) % NUM_BINS;
			smoothHist[j] = (hist[prev] + hist[j] + hist[next]) / 3;
		}
		for (int j = 0; j < NUM_BINS; j++) {
			hist[j] = smoothHist[j];
		}
	}

	// Chọn ra hướng đặc trưng của keypoint
	float ori_thresh = 0.8, ori_max = 0;
	for (int i = 0; i < NUM_BINS; i++) {
		if (hist[i] > ori_max) {
			ori_max = hist[i];
		}
	}

	for (int i = 0; i < NUM_BINS; i++) {
		if (hist[i] >= ori_thresh * ori_max) {
			int prev_idx = (i - 1 + NUM_BINS) % NUM_BINS;
			int next_idx = (i + 1) % NUM_BINS;
			float prev = hist[prev_idx], next = hist[next_idx];
			if (prev >= hist[i] || next >= hist[i]) {
				continue;
			}
			float theta = 2 * PI * i / NUM_BINS + PI / NUM_BINS * (prev - next) / (prev - 2 * hist[i] + next);
			orientations.push_back(theta);
		}
	}
	return orientations;
}
void HandleDescriptorofHarris(Keypoint& kp, float theta, const Gradient& grad) {
	float pix_dist = 1;
	float lambda_desc = 6;;
	float lambda_ori = 1.5;
	kp.sigma = 1;
	float histograms[NUM_HIST][NUM_HIST][NUM_ORI] = { 0 };

	// Tính kích thước của description patch
	float half_size = sqrt(2) * lambda_desc * kp.sigma * (NUM_HIST + 1.) / NUM_HIST;
	int x_start = round((kp.x - half_size) / pix_dist);
	int x_end = round((kp.x + half_size) / pix_dist);
	int y_start = round((kp.y - half_size) / pix_dist);
	int y_end = round((kp.y + half_size) / pix_dist);

	float cos_t = cos(theta), sin_t = sin(theta);
	float patch_sigma = lambda_desc * kp.sigma;

	for (int m = x_start; m <= x_end; m++) {
		for (int n = y_start; n <= y_end; n++) {
			float x = ((m * pix_dist - kp.x) * cos_t
				+ (n * pix_dist - kp.y) * sin_t) / kp.sigma;
			float y = (-(m * pix_dist - kp.x) * sin_t
				+ (n * pix_dist - kp.y) * cos_t) / kp.sigma;

			// Kiểm tra (x, y) có thuộc description patch
			if (max(abs(x), abs(y)) > lambda_desc * (NUM_HIST + 1.) / NUM_HIST)
				continue;

			float gx = GetValueMatrix(grad.Gx, m, n);
			float gy = GetValueMatrix(grad.Gy, m, n);
			float theta_mn = fmod(atan2(gy, gx) - theta + 2 * PI, 2 * PI);
			float grad_norm = sqrt(gx * gx + gy * gy);
			float weight = exp(-(pow(m * pix_dist - kp.x, 2) + pow(n * pix_dist - kp.y, 2))
				/ (2 * patch_sigma * patch_sigma));
			float contribution = weight * grad_norm;

			UpdateHistograms(histograms, x, y, contribution, theta_mn, lambda_desc);
		}
	}

	// Xây dựng mảng description
	int size = NUM_HIST * NUM_HIST * NUM_ORI;
	float* hist = reinterpret_cast<float*>(histograms);

	float norm = 0;
	for (int i = 0; i < size; i++) {
		norm += hist[i] * hist[i];
	}
	norm = sqrt(norm);
	float norm2 = 0;
	for (int i = 0; i < size; i++) {
		hist[i] = min(hist[i], 0.2f * norm);
		norm2 += hist[i] * hist[i];
	}
	norm2 = sqrt(norm2);
	kp.descriptor.resize(128);
	for (int i = 0; i < size; i++) {
		float val = floor(512 * hist[i] / norm2);
		kp.descriptor[i] = (min((int)val, 255));
	}
}

// Bước xử lý orientation và description theo thuật toán Blob
vector<float> HandleOrientationOfBlob(Keypoint& kp, const vector<Gradient>& list_grad) {
	float pix_dist = 1;
	float lambda_ori = 1.5;
	float lambda_desc = 6;
	vector<float> orientations;

	const vector<vector<float>>& gradX = list_grad[kp.scale].Gx;
	const vector<vector<float>>& gradY = list_grad[kp.scale].Gy;
	int rows = gradX.size(), cols = gradX[0].size();

	// Kiểm tra điểm kp có nằm gần biên ảnh không
	float min_dist_from_border = min({ kp.x, kp.y, pix_dist * cols - kp.x, pix_dist * rows - kp.y });
	if (min_dist_from_border <= sqrt(2) * lambda_desc * kp.sigma) {
		return orientations;
	}

	float hist[NUM_BINS] = { 0 };
	int bin;
	float gx, gy, grad_norm, weight, theta;
	float patch_sigma = lambda_ori * kp.sigma;
	float patch_radius = 3 * patch_sigma;
	int x_start = round((kp.x - patch_radius) / pix_dist);
	int x_end = round((kp.x + patch_radius) / pix_dist);
	int y_start = round((kp.y - patch_radius) / pix_dist);
	int y_end = round((kp.y + patch_radius) / pix_dist);

	// Tính toán orientation histogram
	for (int x = x_start; x <= x_end; x++) {
		for (int y = y_start; y <= y_end; y++) {
			gx = GetValueMatrix(gradX, x, y);
			gy = GetValueMatrix(gradY, x, y);
			grad_norm = sqrt(gx * gx + gy * gy);
			weight = exp(-(pow(x * pix_dist - kp.x, 2) + pow(y * pix_dist - kp.y, 2))
				/ (2 * patch_sigma * patch_sigma));
			theta = fmod(atan2(gy, gx) + 2 * PI, 2 * PI);
			bin = (int)round(NUM_BINS / (2 * PI) * theta) % NUM_BINS;
			hist[bin] += weight * grad_norm;
		}
	}

	// Thực hiện làm trơn hist, thực hiện 6 lần bằng filter [1, 1, 1] / 3
	float smoothHist[NUM_BINS];
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < NUM_BINS; j++) {
			int prev = (j - 1 + NUM_BINS) % NUM_BINS;
			int next = (j + 1) % NUM_BINS;
			smoothHist[j] = (hist[prev] + hist[j] + hist[next]) / 3;
		}
		for (int j = 0; j < NUM_BINS; j++) {
			hist[j] = smoothHist[j];
		}
	}

	// Chọn ra hướng đặc trưng của keypoint
	float ori_thresh = 0.8, ori_max = 0;
	for (int i = 0; i < NUM_BINS; i++) {
		if (hist[i] > ori_max) {
			ori_max = hist[i];
		}
	}
	
	for (int i = 0; i < NUM_BINS; i++) {
		if (hist[i] >= ori_thresh * ori_max) {
			int prev_idx = (i - 1 + NUM_BINS) % NUM_BINS;
			int next_idx = (i + 1) % NUM_BINS;
			float prev = hist[prev_idx], next = hist[next_idx];
			if (prev >= hist[i] || next >= hist[i]){
				continue;
			}
			float theta = 2 * PI * i / NUM_BINS + PI / NUM_BINS * (prev - next) / (prev - 2 * hist[i] + next);
			orientations.push_back(theta);
		}
	}
	return orientations;
}
void HandleDescriptorofBlob(Keypoint& kp, float theta, const vector<Gradient>& list_grad) {
	float pix_dist = 1;
	float lambda_desc = 6;;
	float lambda_ori = 1.5;
	float histograms[NUM_HIST][NUM_HIST][NUM_ORI] = { 0 };

	const vector<vector<float>>& gradX = list_grad[kp.scale].Gx;
	const vector<vector<float>>& gradY = list_grad[kp.scale].Gy;

	// Tính kích thước của description patch
	float half_size = sqrt(2) * lambda_desc * kp.sigma * (NUM_HIST + 1.) / NUM_HIST;
	int x_start = round((kp.x - half_size) / pix_dist);
	int x_end = round((kp.x + half_size) / pix_dist);
	int y_start = round((kp.y - half_size) / pix_dist);
	int y_end = round((kp.y + half_size) / pix_dist);

	float cos_t = cos(theta), sin_t = sin(theta);
	float patch_sigma = lambda_desc * kp.sigma;

	for (int m = x_start; m <= x_end; m++) {
		for (int n = y_start; n <= y_end; n++) {
			
			float x = ((m * pix_dist - kp.x) * cos_t + (n * pix_dist - kp.y) * sin_t) / kp.sigma;
			float y = (-(m * pix_dist - kp.x) * sin_t + (n * pix_dist - kp.y) * cos_t) / kp.sigma;

			// Kiểm tra (x, y) có thuộc description patch
			if (max(abs(x), abs(y)) > lambda_desc * (NUM_HIST + 1.) / NUM_HIST)
				continue;

			float gx = GetValueMatrix(gradX, m, n);
			float gy = GetValueMatrix(gradY, m, n);
			float theta_mn = fmod(atan2(gy, gx) - theta + 2 * PI, 2 * PI);
			float grad_norm = sqrt(gx * gx + gy * gy);
			float weight = exp(-(pow(m * pix_dist - kp.x, 2) + pow(n * pix_dist - kp.y, 2)) / (2 * patch_sigma * patch_sigma));
			float contribution = weight * grad_norm;

			UpdateHistograms(histograms, x, y, contribution, theta_mn, lambda_desc);
		}
	}

	// Xây dựng mảng description
	int size = NUM_HIST * NUM_HIST * NUM_ORI;
	float* hist = reinterpret_cast<float*>(histograms);

	float norm = 0;
	for (int i = 0; i < size; i++) {
		norm += hist[i] * hist[i];
	}
	norm = sqrt(norm);
	float norm2 = 0;
	for (int i = 0; i < size; i++) {
		hist[i] = min(hist[i], 0.2f * norm);
		norm2 += hist[i] * hist[i];
	}
	norm2 = sqrt(norm2);
	kp.descriptor.resize(128);
	for (int i = 0; i < size; i++) {
		float val = floor(512 * hist[i] / norm2);
		kp.descriptor[i] = (min((int)val, 255));
	}
}

// Bước xử lý orientation và description theo thuật toán DoG
vector<float> HandleOrientationOfDoG(Keypoint& kp, const ScaleSpacePyramid& grad_pyramid) {
	float pix_dist = 0.5 * pow(2, kp.octave);
	float lambda_ori = 1.5;
	float lambda_desc = 6;
	vector<float> orientations;

	const vector<vector<float>>& gradX = grad_pyramid.gradients[kp.octave][kp.scale].Gx;
	const vector<vector<float>>& gradY = grad_pyramid.gradients[kp.octave][kp.scale].Gy;
	int rows = gradX.size(), cols = gradX[0].size();

	// Kiểm tra điểm kp có nằm gần biên ảnh không
	float min_dist_from_border = min({ kp.x, kp.y, pix_dist * cols - kp.x, pix_dist * rows - kp.y });
	if (min_dist_from_border <= sqrt(2) * lambda_desc * kp.sigma) {
		return orientations;
	}

	float hist[NUM_BINS] = { };
	int bin;
	float gx, gy, grad_norm, weight, theta;
	float patch_sigma = lambda_ori * kp.sigma;
	float patch_radius = 3 * patch_sigma;
	int x_start = round((kp.x - patch_radius) / pix_dist);
	int x_end = round((kp.x + patch_radius) / pix_dist);
	int y_start = round((kp.y - patch_radius) / pix_dist);
	int y_end = round((kp.y + patch_radius) / pix_dist);

	// Tính toán orientation histogram
	for (int x = x_start; x <= x_end; x++) {
		for (int y = y_start; y <= y_end; y++) {
			gx = GetValueMatrix(gradX, x, y);
			gy = GetValueMatrix(gradY, x, y);
			grad_norm = sqrt(gx * gx + gy * gy);
			weight = exp(-(pow(x * pix_dist - kp.x, 2) + pow(y * pix_dist - kp.y, 2)) / (2 * patch_sigma * patch_sigma));
			theta = fmod(atan2(gy, gx) + 2 * PI, 2 * PI);
			bin = (int)round(NUM_BINS / (2 * PI) * theta) % NUM_BINS;
			hist[bin] += weight * grad_norm;
		}
	}

	// Thực hiện làm trơn hist, thực hiện 6 lần bằng filter [1, 1, 1] / 3
	float smoothHist[NUM_BINS];
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < NUM_BINS; j++) {
			int prev = (j - 1 + NUM_BINS) % NUM_BINS;
			int next = (j + 1) % NUM_BINS;
			smoothHist[j] = (hist[prev] + hist[j] + hist[next]) / 3;
		}
		for (int j = 0; j < NUM_BINS; j++) {
			hist[j] = smoothHist[j];
		}
	}

	// Chọn ra hướng đặc trưng của keypoint
	float ori_thresh = 0.8, ori_max = 0;
	for (int i = 0; i < NUM_BINS; i++) {
		if (hist[i] > ori_max) {
			ori_max = hist[i];
		}
	}
	
	for (int i = 0; i < NUM_BINS; i++) {
		if (hist[i] >= ori_thresh * ori_max) {
			int prev_idx = (i - 1 + NUM_BINS) % NUM_BINS;
			int next_idx = (i + 1) % NUM_BINS;
			float prev = hist[prev_idx], next = hist[next_idx];
			if (prev >= hist[i] || next >= hist[i]){
				continue;
			}
			float theta = 2 * PI * i / NUM_BINS + PI / NUM_BINS * (prev - next) / (prev - 2 * hist[i] + next);
			orientations.push_back(theta);
		}
	}
	return orientations;
}
void HandleDescriptorofDoG(Keypoint& kp, float theta, const ScaleSpacePyramid& grad_pyramid) {
	float pix_dist = 0.5 * pow(2, kp.octave);
	float lambda_desc = 6;;
	float lambda_ori = 1.5;
	float histograms[NUM_HIST][NUM_HIST][NUM_ORI] = { 0 };

	const vector<vector<float>>& gradX = grad_pyramid.gradients[kp.octave][kp.scale].Gx;
	const vector<vector<float>>& gradY = grad_pyramid.gradients[kp.octave][kp.scale].Gy;

	// Tính kích thước của description patch
	float half_size = sqrt(2) * lambda_desc * kp.sigma * (NUM_HIST + 1.) / NUM_HIST;
	int x_start = round((kp.x - half_size) / pix_dist);
	int x_end = round((kp.x + half_size) / pix_dist);
	int y_start = round((kp.y - half_size) / pix_dist);
	int y_end = round((kp.y + half_size) / pix_dist);

	float cos_t = cos(theta), sin_t = sin(theta);
	float patch_sigma = lambda_desc * kp.sigma;

	for (int m = x_start; m <= x_end; m++) {
		for (int n = y_start; n <= y_end; n++) {
			float x = ((m * pix_dist - kp.x) * cos_t + (n * pix_dist - kp.y) * sin_t) / kp.sigma;
			float y = (-(m * pix_dist - kp.x) * sin_t + (n * pix_dist - kp.y) * cos_t) / kp.sigma;

			// Kiểm tra (x, y) có thuộc description patch
			if (max(abs(x), abs(y)) <= lambda_desc * (NUM_HIST + 1.) / NUM_HIST)
			{
				float gx = GetValueMatrix(gradX, m, n);
				float gy = GetValueMatrix(gradY, m, n);
				float theta_mn = fmod(atan2(gy, gx) - theta + 2 * PI, 2 * PI);
				float grad_norm = sqrt(gx * gx + gy * gy);
				float weight = exp(-(pow(m * pix_dist - kp.x, 2) + pow(n * pix_dist - kp.y, 2)) / (2 * patch_sigma * patch_sigma));
				float contribution = weight * grad_norm;

				UpdateHistograms(histograms, x, y, contribution, theta_mn, lambda_desc);
			}
		}
	}

	// Xây dựng mảng description
	int size = NUM_HIST * NUM_HIST * NUM_ORI;
	float* hist = reinterpret_cast<float*>(histograms);

	float norm = 0;
	for (int i = 0; i < size; i++) {
		norm += hist[i] * hist[i];
	}
	norm = sqrt(norm);
	float norm2 = 0;
	for (int i = 0; i < size; i++) {
		hist[i] = min(hist[i], 0.2f * norm);
		norm2 += hist[i] * hist[i];
	}
	norm2 = sqrt(norm2);
	kp.descriptor.resize(128);
	for (int i = 0; i < size; i++) {
		float val = floor(512 * hist[i] / norm2);
		kp.descriptor[i] = (min((int)val, 255));
	}
}

// Tổng hợp các bước tính toán của thuật toán SIFT
vector<Keypoint> HandleDetectBySift(const Mat& img, int detector) {
// Bước 1: Detector - có thể chọn 1 trong 3 thuật toán detector
// Bước 2: Description
	vector<Keypoint> kps;
	vector<Keypoint> final_kps;
	if (detector == 0) {
		kps = HandleDetectByHarris(img);
		// Tính Orientation & Description
		Gradient grad = GetGradientOfHarris(img);
		for (int i = 0; i < kps.size(); i++) {
			Keypoint kp = kps[i];
			vector<float> orientations = HandleOrientationOfHarris(kp, grad);
			for (int j = 0; j < orientations.size(); j++) {
				float theta = orientations[j];
				HandleDescriptorofHarris(kp, theta, grad);
				final_kps.push_back(kp);
			}
		}
	}
	else if (detector == 1) {
		kps = HandleDetectByBlob(img);
		// Tính Orientation & Description
		vector<Gradient> list_grad = GetGradientOfBlob(img);
		for (int i = 0; i < kps.size(); i++) {
			Keypoint kp = kps[i];
			vector<float> orientations = HandleOrientationOfBlob(kp, list_grad);
			for (int j = 0; j < orientations.size(); j++) {
				float theta = orientations[j];
				HandleDescriptorofBlob(kp, theta, list_grad);
				final_kps.push_back(kp);
			}
		}
	}
	else if (detector == 2) {
		kps = HandleDetectByDoG(img);
		// Tính Orientation & Description
		ScaleSpacePyramid list_grad = GetGradientOfDoG(img);
		for (int i = 0; i < kps.size(); i++) {
			Keypoint kp = kps[i];
			vector<float> orientations = HandleOrientationOfDoG(kp, list_grad);
			for (int j = 0; j < orientations.size(); j++) {
				float theta = orientations[j];
				HandleDescriptorofDoG(kp, theta, list_grad);
				final_kps.push_back(kp);
			}
		}
	}

	return final_kps;
}

// Hàm kiểm tra các cặp điểm khớp với nhau
vector<pair<int, int>> FindKeypointMatches(	const vector<Keypoint>& kps_a, const vector<Keypoint>& kps_b, int detector)
{
	// Chuyển vector<Keypoint> sang Mat
	Mat descriptors_a(kps_a.size(), 128, CV_32F), descriptor_b(kps_b.size(), 128, CV_32F);

	for (int i = 0; i < kps_a.size(); i++) {
		for (int j = 0; j < 128; j++) {
			descriptors_a.at<float>(i, j) = kps_a[i].descriptor[j];
		}
	}
	for (int i = 0; i < kps_b.size(); i++) {
		for (int j = 0; j < 128; j++) {
			descriptor_b.at<float>(i, j) = kps_b[i].descriptor[j];
		}
	}

	// Sử dụng KNN
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptors_a, descriptor_b, knn_matches, 3);

	vector<pair<int, int>> matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (detector == 0 && knn_matches[i][0].distance < 0.83 * knn_matches[i][1].distance) {
			matches.push_back({ knn_matches[i][0].queryIdx, knn_matches[i][0].trainIdx });
		}
		else if (detector == 1 && knn_matches[i][0].distance < 0.65 * knn_matches[i][1].distance) {
			matches.push_back({ knn_matches[i][0].queryIdx, knn_matches[i][0].trainIdx });
		}
		else if (detector == 2 && knn_matches[i][0].distance < 0.5 * knn_matches[i][1].distance) {
			matches.push_back({ knn_matches[i][0].queryIdx, knn_matches[i][0].trainIdx });
		}
	}
	return matches;
}
Mat DrawMatches(Mat img_a, Mat img_b, vector<Keypoint> kps_a, vector<Keypoint> kps_b, vector<pair<int, int>> matches)
{
	int a_rows = img_a.rows, a_cols = img_a.cols;
	int b_rows = img_b.rows, b_cols = img_b.cols;
	Mat res(cv::Size(a_cols + b_cols, max(a_rows, b_rows)), img_a.type(), cv::Scalar::all(0));

	for (int y = 0; y < a_rows; y++) {
		for (int x = 0; x < a_cols; x++) {
			res.at<cv::Vec3b>(y, x)[0] = img_a.at<cv::Vec3b>(y, x)[0];
			res.at<cv::Vec3b>(y, x)[1] = img_a.at<cv::Vec3b>(y, x)[img_a.channels() == 3 ? 1 : 0];
			res.at<cv::Vec3b>(y, x)[2] = img_a.at<cv::Vec3b>(y, x)[img_a.channels() == 3 ? 2 : 0];
		}
	}

	for (int y = 0; y < b_rows; y++) {
		for (int x = 0; x < b_cols; x++) {
			res.at<cv::Vec3b>(y, x + a_cols)[0] = img_b.at<cv::Vec3b>(y, x)[0];
			res.at<cv::Vec3b>(y, x + a_cols)[1] = img_b.at<cv::Vec3b>(y, x)[img_b.channels() == 3 ? 1 : 0];
			res.at<cv::Vec3b>(y, x + a_cols)[2] = img_b.at<cv::Vec3b>(y, x)[img_b.channels() == 3 ? 2 : 0];
		}
	}

	vector<cv::Scalar> colors = {
	cv::Scalar(240, 88, 66),
	cv::Scalar(20,100,250),
	cv::Scalar(20, 220, 100),
	};

	for (int i = 0; i < matches.size(); i++) {
		cv::Point2f pt_a(kps_a[matches[i].first].x, kps_a[matches[i].first].y);
		cv::Point2f pt_b(kps_b[matches[i].second].x + a_cols, kps_b[matches[i].second].y);
		cv::Scalar color = colors[i % colors.size()];
		circle(res, pt_a, 5, color, 1, LINE_AA);
		circle(res, pt_b, 5, color, 1, LINE_AA);
		cv::line(res, pt_a, pt_b, color, 1, LINE_AA);
	}

	return res;
}

double matchBySIFT(const Mat& img1, const Mat& img2, int detector) {
	vector<Keypoint> kps_a = HandleDetectBySift(img1, detector);
	vector<Keypoint> kps_b = HandleDetectBySift(img2, detector);

	cout << "The number of feature points of the image 1: " << kps_a.size() << endl;
	cout << "The number of feature points of the image 2: " << kps_b.size() << endl;

	vector<pair<int, int>> matches = FindKeypointMatches(kps_a, kps_b, detector);
	cout << "Number of matches: " << matches.size() << endl;

	Mat res = DrawMatches(img1, img2, kps_a, kps_b, matches);

	string title = "Matches image by Sift";

	if (detector == 0) {
		title = title + " with detector Harris";
	}
	else if (detector == 1) {
		title = title + " with detector Blob";
	}
	else if (detector == 2) {
		title = title + " with detector Dog";
	}

	ShowImage(res, title);

	return 0;
}

// main
// ----------------------------
int main(int argc, char** argv) {

	if (argc != 3 && argc != 5) {
		cout << "Invalid input parameter.\n";
		return 0;
	}

	// Sử dụng 3 thuật toán để xác định điểm đặc trưng
	if (argc == 3) {
		Mat img = imread(argv[1]);
		if (!img.data) {
			cout << "Can't read image.\n";
			return 0;
		}

		if (!strcmp(argv[2], "harris")) {
			Mat res = detectHarris(img);
			ShowImage(res, "Feature Points by Harris");
		}
		else if (!strcmp(argv[2], "blob")) {
			Mat res = detectBlob(img);
			ShowImage(res, "Feature Points by Blob");
		}
		else if (!strcmp(argv[2], "dog")) {
			Mat res = detectDog(img);
			ShowImage(res, "Feature Points by Dog");
		}
		
	}
	// Sử dụng thuật toán Sift
	if (argc == 5 && !strcmp(argv[3], "sift")) {
		Mat img_a = imread(argv[1]);
		if (!img_a.data) {
			cout << "Can't read image 1.\n";
			return 0;
		}
		Mat img_b = imread(argv[2]);
		if (!img_b.data) {
			cout << "Can't read image 2.\n";
			return 0;
		}

		int detector = atoi(argv[4]);
		if (detector == 1) {
			cout << "Use Harris detection.\n";
			matchBySIFT(img_a, img_b, 0);
		}
		else if (detector == 2) {
			cout << "Use Blob detection.\n";
			matchBySIFT(img_a, img_b, 1);
		}
		else if (detector == 3) {
			cout << "Use DoG detection.\n";
			matchBySIFT(img_a, img_b, 2);
		}
	}

	waitKey(0);
	destroyAllWindows();
	return 1;
}

