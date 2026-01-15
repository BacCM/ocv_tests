#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>

using namespace cv;
using namespace dnn;
using namespace std;

// Структура для хранения результатов детекции
struct Detection {
	int classId;
	float confidence;
	Rect box;
	string className;
};

class StationeryDetector {
private:
	Net net;
	vector<string> classes;
	float confThreshold = 0.5f;
	float nmsThreshold = 0.4f;
	int inputWidth = 416;
	int inputHeight = 416;

public:
	// Конструктор - загружает модель и классы
	StationeryDetector(const string& modelConfig,
		const string& modelWeights,
		const string& classesFile) {
		// Загружаем модель YOLO
		net = readNetFromDarknet(modelConfig, modelWeights);
		net.setPreferableBackend(DNN_BACKEND_OPENCV);
		net.setPreferableTarget(DNN_TARGET_CPU);

		// Загружаем названия классов
		ifstream ifs(classesFile.c_str());
		string line;
		while (getline(ifs, line)) {
			classes.push_back(line);
		}

		cout << "Модель загружена. Количество классов: " << classes.size() << endl;
	}

	// Обновление параметров
	void setConfidenceThreshold(float threshold) { confThreshold = threshold; }
	void setNMSThreshold(float threshold) { nmsThreshold = threshold; }
	void setInputSize(int width, int height) {
		inputWidth = width;
		inputHeight = height;
	}

	// Функция детекции
	vector<Detection> detect(Mat& frame) {
		vector<Detection> detections;

		// Подготовка изображения для нейронной сети
		Mat blob;
		blobFromImage(frame, blob, 1 / 255.0, Size(inputWidth, inputHeight),
			Scalar(0, 0, 0), true, false);

		// Устанавливаем blob на вход сети
		net.setInput(blob);

		// Получаем выходы сети
		vector<Mat> outputs;
		vector<string> outputLayerNames = net.getUnconnectedOutLayersNames();
		net.forward(outputs, outputLayerNames);

		// Обработка выходов
		processOutputs(frame, outputs, detections);

		// Применяем Non-Maximum Suppression
		applyNMS(detections);

		return detections;
	}

private:
	void processOutputs(Mat& frame, vector<Mat>& outputs, vector<Detection>& detections) {
		vector<int> classIds;
		vector<float> confidences;
		vector<Rect> boxes;

		for (const auto& output : outputs) {
			float* data = (float*)output.data;

			for (int i = 0; i < output.rows; ++i) {
				// Пропускаем первые 4 значения (bbox координаты)
				// Следующие значения - confidence для каждого класса
				Mat scores = output.row(i).colRange(5, output.cols);
				Point classIdPoint;
				double confidence;

				minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);

				if (confidence > confThreshold) {
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(Rect(left, top, width, height));
				}

				data += output.cols;
			}
		}

		// Сохраняем детекции
		for (size_t i = 0; i < boxes.size(); ++i) {
			Detection detection;
			detection.classId = classIds[i];
			detection.confidence = confidences[i];
			detection.box = boxes[i];
			detection.className = classes[classIds[i]];
			detections.push_back(detection);
		}
	}

	void applyNMS(vector<Detection>& detections) {
		vector<int> indices;
		vector<Rect> boxes;
		vector<float> scores;

		for (const auto& det : detections) {
			boxes.push_back(det.box);
			scores.push_back(det.confidence);
		}

		NMSBoxes(boxes, scores, confThreshold, nmsThreshold, indices);

		// Оставляем только детекции, прошедшие NMS
		vector<Detection> filteredDetections;
		for (int idx : indices) {
			filteredDetections.push_back(detections[idx]);
		}

		detections = filteredDetections;
	}
};

// Функция для отрисовки результатов
void drawDetections(Mat& frame, const vector<Detection>& detections) {
	// Палитра цветов для разных классов
	vector<Scalar> colors = {
		Scalar(255, 0, 0),   // синий
		Scalar(0, 255, 0),   // зеленый
		Scalar(0, 0, 255),   // красный
		Scalar(255, 255, 0), // голубой
		Scalar(255, 0, 255), // фиолетовый
		Scalar(0, 255, 255), // желтый
		Scalar(128, 0, 128), // пурпурный
		Scalar(0, 128, 128)  // оливковый
	};

	for (const auto& detection : detections) {
		int classId = detection.classId % colors.size();
		Scalar color = colors[classId];

		// Рисуем прямоугольник
		rectangle(frame, detection.box, color, 2);

		// Создаем подпись
		string label = format("%s: %.2f", detection.className.c_str(), detection.confidence);

		// Рисуем фон для текста
		int baseline;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
		rectangle(frame,
			Point(detection.box.x, detection.box.y - labelSize.height - 5),
			Point(detection.box.x + labelSize.width, detection.box.y),
			color, FILLED);

		// Рисуем текст
		putText(frame, label,
			Point(detection.box.x, detection.box.y - 5),
			FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
	}

	// Выводим статистику
	string stats = format("Обнаружено: %zu предметов", detections.size());
	putText(frame, stats, Point(10, 30),
		FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
}

int main() {
	// Пути к файлам модели
	string modelConfig = "yolov3-tiny.cfg";
	string modelWeights = "yolov3-tiny.weights";
	string classesFile = "stationery.names";

	// Создаем детектор
	StationeryDetector detector(modelConfig, modelWeights, classesFile);
	detector.setConfidenceThreshold(0.5);
	detector.setNMSThreshold(0.4);

	// Открываем видео или камеру
///	VideoCapture cap(0); // Для камеры
	VideoCapture cap("t2.jpg"); // Для видео файла

	if (!cap.isOpened()) {
		cerr << "Ошибка открытия камеры!" << endl;
		return -1;
	}

	// Создаем окна
	namedWindow("Stationery Detection", WINDOW_NORMAL);
	namedWindow("Detection Info", WINDOW_NORMAL);

	// Для измерения FPS
	auto start = chrono::steady_clock::now();
	int frameCount = 0;

	while (true) {
		Mat frame;
		cap >> frame;

		if (frame.empty()) {
			cerr << "Пустой кадр!" << endl;
			break;
		}

		// Детекция предметов
		auto startDetection = chrono::steady_clock::now();
		vector<Detection> detections = detector.detect(frame);
		auto endDetection = chrono::steady_clock::now();

		// Отрисовка результатов
		Mat displayFrame = frame.clone();
		drawDetections(displayFrame, detections);

		// Вывод информации о детекциях
		Mat infoFrame(300, 400, CV_8UC3, Scalar(50, 50, 50));
		int yPos = 30;

		for (const auto& det : detections) {
			string info = format("%s: %.1f%%", det.className.c_str(), det.confidence * 100);
			putText(infoFrame, info, Point(10, yPos),
				FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 100), 1);
			yPos += 25;
		}

		// Вывод FPS
		frameCount++;
		auto end = chrono::steady_clock::now();
		chrono::duration<double> elapsed = end - start;

		if (elapsed.count() >= 1.0) {
			double fps = frameCount / elapsed.count();
			string fpsText = format("FPS: %.1f", fps);
			putText(displayFrame, fpsText, Point(10, 60),
				FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

			// Время детекции
			chrono::duration<double> detectionTime = endDetection - startDetection;
			string timeText = format("Detection: %.0f ms", detectionTime.count() * 1000);
			putText(displayFrame, timeText, Point(10, 90),
				FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

			frameCount = 0;
			start = chrono::steady_clock::now();
		}

		// Отображение
		imshow("Stationery Detection", displayFrame);
		imshow("Detection Info", infoFrame);

		// Выход по ESC
		if (waitKey(1) == 27) {
			break;
		}
	}
	for (;;) {
		if (waitKey(1) == 27) {
			break;
		}
	}
	cap.release();
	destroyAllWindows();

	return 0;
}