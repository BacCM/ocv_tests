#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main() 
{
	std::setlocale(LC_ALL, "ru_RU.UTF8");

	vector<string> classes;
	string classesFile = "coco.names";

	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) {
		classes.push_back(line);
	}

	// Загружаем модель YOLO
	String modelConfiguration = "yolov3.cfg";
	String modelWeights = "yolov3.weights";

	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Загружаем изображение
	Mat frame = imread("h1.jpg");
	if (frame.empty()) {
		cout << "Не удалось загрузить изображение!" << endl;
		return -1;
	}

	// Подготавливаем изображение для нейронной сети
	Mat blob;
	double scale = 1.0 / 255.0;
	Size size = Size(416, 416);
	Scalar mean = Scalar(0, 0, 0);
	bool swapRB = true;
	bool crop = false;

	blobFromImage(frame, blob, scale, size, mean, swapRB, crop);

	// Устанавливаем blob как вход нейронной сети
	net.setInput(blob);

	// Получаем выходные слои
	vector<String> outNames = net.getUnconnectedOutLayersNames();
	vector<Mat> outs;
	net.forward(outs, outNames);

	// Параметры для фильтрации результатов
	float confThreshold = 0.5; // Порог уверенности
	float nmsThreshold = 0.4;  // Порог для Non-Maximum Suppression

	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	// Обрабатываем выходы нейронной сети
	for (size_t i = 0; i < outs.size(); ++i) {
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;

			// Находим класс с максимальной уверенностью
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

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
		}
	}

	// Применяем Non-Maximum Suppression для удаления дублирующих боксов
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	// Рисуем результаты на изображении
	for (size_t i = 0; i < indices.size(); ++i) {
		int idx = indices[i];
		Rect box = boxes[idx];

		// Рисуем прямоугольник вокруг объекта
		rectangle(frame, box, Scalar(0, 255, 0), 2);

		// Добавляем подпись с классом и уверенностью
		string label = format("%s: %.2f", classes[classIds[idx]].c_str(), confidences[idx]);

		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		int top = max(box.y, labelSize.height);

		rectangle(frame, Point(box.x, top - labelSize.height),
			Point(box.x + labelSize.width, top + baseLine),
			Scalar(0, 255, 0), FILLED);
		putText(frame, label, Point(box.x, top),
			FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
	}

	// Сохраняем и показываем результат
	imwrite("output.jpg", frame);
	imshow("Object Detection", frame);
	waitKey(0);

	return 0;
}