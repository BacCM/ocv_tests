#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>


#include <iostream>

using namespace cv;
using namespace std;

int main() {
	// Захват видео с камеры
/*	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "Камера не найдена!" << endl;
		return -1;
	}
*/
	Mat image = imread("test.jpg");

	if (image.empty()) {
		cout << "Не удалось загрузить имадж!" << endl;
	}

	// Загрузка каскада для детектирования лиц
	CascadeClassifier faceCascade;
	if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
		cout << "Не удалось загрузить каскад!" << endl;
		// Можно скачать с: https://github.com/opencv/opencv/tree/master/data/haarcascades
		return -1;
	}

	Mat frame = image;
	while (true) {
//		cap >> frame;
		if (frame.empty()) break;

		// Конвертируем в оттенки серого для детектора
		Mat gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		equalizeHist(gray, gray);

		// Детектируем лица
		vector<Rect> faces;
		faceCascade.detectMultiScale(gray, faces, 1.1, 3);

		// Рисуем прямоугольники вокруг лиц
		for (const Rect& face : faces) {
			rectangle(frame, face, Scalar(0, 255, 0), 2);
			putText(frame, "Face", Point(face.x, face.y - 5),
				FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
		}

		// Показываем результат
		imshow("Face Detection", frame);

		// Выход по нажатию 'q'
		if (waitKey(10) == 'q') break;
	}

	return 0;
}