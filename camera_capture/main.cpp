#include <opencv2/opencv.hpp>
#include <iostream>
#include <locale>

int main() 
{
	std::setlocale(LC_ALL, "ru_RU.UTF8");
	// Открываем камеру (0 - индекс камеры по умолчанию)
	cv::VideoCapture cap(0);

	// Проверяем, открылась ли камера
	if (!cap.isOpened()) {
		std::cerr << "Ошибка: не удалось открыть камеру!" << std::endl;
		return -1;
	}

	// Создаем окно для отображения видео
	cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);

	cv::Mat frame;

	while (true) {
		// Захватываем кадр
		cap >> frame;

		// Если кадр пустой, выходим из цикла
		if (frame.empty()) {
			break;
		}

		// Отображаем кадр
		cv::imshow("Webcam", frame);

		// Выходим по нажатию клавиши 'q' или ESC
		char key = cv::waitKey(30);
		if (key == 'q' || key == 27) { // 27 - код клавиши ESC
			break;
		}
	}

	// Освобождаем ресурсы
	cap.release();
	cv::destroyAllWindows();

	return 0;
}