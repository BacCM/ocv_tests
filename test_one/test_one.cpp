#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
	// Пример 1: Загрузка и отображение изображения
	Mat image = imread("test.jpg");

	if (image.empty()) {
		cout << "Не удалось загрузить изображение!" << endl;
		// Создаем тестовое изображение
		image = Mat(300, 400, CV_8UC3, Scalar(100, 150, 200));
		putText(image, "OpenCV Test", Point(50, 150),
			FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
	}

	namedWindow("Test Window", WINDOW_NORMAL);
	imshow("Test Window", image);

	// Пример 2: Преобразование в оттенки серого
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	imshow("Gray Image", gray);

	// Пример 3: Определение границ
	Mat edges;
	Canny(gray, edges, 50, 150);
	imshow("Edges", edges);

	waitKey(0);
	return 0;
}