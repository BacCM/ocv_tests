#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Структура для хранения информации о распознанном предмете
struct StationeryItem {
	string type;
	Rect boundingBox;
	Scalar color;
};

// Функция для определения типа предмета по характеристикам
string classifyItem(const vector<Point>& contour, const Mat& frame) {
	double area = contourArea(contour);
	RotatedRect rotatedRect = minAreaRect(contour);
	float aspectRatio = (float)rotatedRect.size.width / rotatedRect.size.height;

	// Фильтруем слишком маленькие объекты
	if (area < 500) return "unknown";

	// Определяем форму по отношению сторон
	if (aspectRatio > 2.5 || aspectRatio < 0.4) {
		// Длинные и тонкие объекты - вероятно ручки/карандаши
		if (area < 1500) return "pen/pencil";
		else return "ruler";
	}
	else if (aspectRatio > 0.8 && aspectRatio < 1.2) {
		// Более квадратные объекты
		if (area > 5000) return "notebook";
		else return "eraser";
	}
	else if (area > 10000) {
		return "book";
	}

	return "stationery_item";
}

int main()
{

	std::setlocale(LC_ALL, "ru_RU.UTF8");
	// Открываем камеру или загружаем изображение
//	VideoCapture cap(0); // 0 - индекс камеры, или укажите путь к файлу
	VideoCapture cap("t2.jpg"); // для работы с изображением

	if (!cap.isOpened()) {
		cerr << "Ошибка открытия камеры!" << endl;
		return -1;
	}

	namedWindow("Original", WINDOW_NORMAL);
	namedWindow("Processed", WINDOW_NORMAL);
	namedWindow("Objects", WINDOW_NORMAL);

	while (true) {
		Mat frame;
		cap >> frame;

		if (frame.empty()) {
			cerr << "Пустой кадр!" << endl;
			break;
		}

		// 1. Предварительная обработка
		Mat blurred, hsv, processed;
		GaussianBlur(frame, blurred, Size(5, 5), 0);
		cvtColor(blurred, hsv, COLOR_BGR2HSV);

		// 2. Создаем маску для выделения объектов
		Mat mask;
		// Пример: выделение синих объектов (ручки, карандаши)
		Mat blueMask;
		inRange(hsv, Scalar(100, 50, 50), Scalar(130, 255, 255), blueMask);

		// Пример: выделение красных объектов
		Mat redMask1, redMask2, redMask;
		inRange(hsv, Scalar(0, 50, 50), Scalar(10, 255, 255), redMask1);
		inRange(hsv, Scalar(170, 50, 50), Scalar(180, 255, 255), redMask2);
		redMask = redMask1 | redMask2;

		// Пример: выделение желтых/оранжевых объектов (ластики)
		Mat yellowMask;
		inRange(hsv, Scalar(20, 50, 50), Scalar(30, 255, 255), yellowMask);

		// Объединяем маски
		mask = blueMask | redMask | yellowMask;

		// Морфологические операции для улучшения маски
		Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
		morphologyEx(mask, mask, MORPH_CLOSE, kernel);
		morphologyEx(mask, mask, MORPH_OPEN, kernel);

		// 3. Находим контуры
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		// 4. Анализ и классификация контуров
		vector<StationeryItem> detectedItems;
		Mat objectsDisplay = frame.clone();

		for (size_t i = 0; i < contours.size(); i++) {
			double area = contourArea(contours[i]);

			// Пропускаем слишком маленькие контуры
			if (area < 500) continue;

			// Получаем ограничивающий прямоугольник
			Rect boundingRect = cv::boundingRect(contours[i]);

			// Классифицируем предмет
			string itemType = classifyItem(contours[i], frame);

			// Создаем структуру с информацией о предмете
			StationeryItem item;
			item.type = itemType;
			item.boundingBox = boundingRect;

			// Определяем доминирующий цвет в области
			Mat roi = frame(boundingRect);
			Scalar meanColor = mean(roi);
			item.color = meanColor;

			detectedItems.push_back(item);

			// Рисуем контур и информацию
			drawContours(objectsDisplay, contours, (int)i, Scalar(0, 255, 0), 2);
			rectangle(objectsDisplay, boundingRect, Scalar(255, 0, 0), 2);

			// Выводим текст с типом предмета
			putText(objectsDisplay, itemType,
				Point(boundingRect.x, boundingRect.y - 10),
				FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
		}

		// 5. Вывод информации в консоль
		cout << "Обнаружено предметов: " << detectedItems.size() << endl;
		for (const auto& item : detectedItems) {
			cout << "Тип: " << item.type
				<< ", Позиция: (" << item.boundingBox.x << ", " << item.boundingBox.y
				<< "), Размер: " << item.boundingBox.width << "x" << item.boundingBox.height << endl;
		}

		// 6. Отображение результатов
		imshow("Original", frame);
		imshow("Processed", mask);
		imshow("Objects", objectsDisplay);

		// Выход по нажатию ESC
		if (waitKey(30) == 27) {
			break;
		}
	}

	for (;;) {
		if (waitKey(30) == 27) {
			break;
		}
	}

	cap.release();
	destroyAllWindows();

	return 0;
}