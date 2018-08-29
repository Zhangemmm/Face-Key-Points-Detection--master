///////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// 增强版/////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
/***********************************************
*Date:2018.8.9
*author=syy
*函数功能：人脸关键点回归预测
*输入参数：1.网络模型Deploy 2.训练好的网络模型caffemodel 3.验证集list地址
*         4.验证图片地址 5.预测值输出地址（包括文件名) 6.标注好的图片保存地址
*输出：1.验证集预测值文件 2.标注好的图片
*Version：v1.3 ：1.输入标注文件和图片直接给出预测值和标注好的图片
2.实现连续多张图片的预测
***********************************************/

#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
//#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <sstream>//string转换float头文件
//#include <iomanip>//控制输出的小数点位数<< setprecision(3)
#ifdef USE_OPENCV
using namespace caffe;// NOLINT(build/namespaces)
//避免与caffe命名空间冲突
using std::string;
using std::vector;
using std::ifstream;
using std::istringstream;
/* Pair (label, confidence) representing a prediction. */
class Regressioner {
public:
	Regressioner(const string& model_file, //构造函数声明，网络模型，训练好的model
		const string& trained_file);//const string& mean_file 去掉均值输入
	std::vector<float> Regression(const cv::Mat& img);	//Regressio,对输入的图形
private:
	std::vector<float> Predict(const cv::Mat& img);//predict函数调用process函数将图片输入网络中，使用net_->Forward()函数进行预测											 
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
private://类的私有变量
	shared_ptr<Net<float>> net_;//网络
	cv::Size input_geometry_;//输入层图像大小
	int num_channels_;//输入通道数
};

Regressioner::Regressioner(const string& model_file, const string& trained_file) {
	Caffe::set_mode(Caffe::GPU);
	net_.reset(new Net<float>(model_file, TEST));//加载配置文件 ，模式为test
	net_->CopyTrainedLayersFrom(trained_file);//加载caffemodel，该函数再net.cpp中实现  //要求输入输出都是1指的是blob个数
	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)//判断通道数
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	Blob<float>* output_layer = net_->output_blobs()[0];// 输出层只有一个Blob，因此用[0]; 另外，输出层的shape为（1, 10）
}

std::vector<float> Regressioner::Regression(const cv::Mat& img) {
	std::vector<float> output = Predict(img);// 调用Predict函数对输入图像进行预测
	return output;
}

std::vector<float> Regressioner::Predict(const cv::Mat& img) {  //输入形参为单张图像 
	Blob<float>* input_layer = net_->input_blobs()[0];//网络输入层
	input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);//调整输入图片尺寸
																						  /* Forward dimension change to all layers. */
	net_->Reshape();
	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);	// 为了获得net_网络的输入层数据的指针，直接把输入图片数据拷贝到这个指针里面
	Preprocess(img, &input_channels);	// 输入带预测的图片数据，然后进行预处理，包括归一化、缩放等操作  
	net_->Forward();//前项传播
	Blob<float>* output_layer = net_->output_blobs()[0];/* Copy the output layer to a std::vector */
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}
void Regressioner::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];//网络输入

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Regressioner::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {// 图片预处理函数，包括图片缩放、归一化、3通道图片分开存储
																						 /* Convert the input image to the input image format of the network. */
	cv::Mat sample;//根据通道设置图片颜色
				   //std::cout << "img.channels()" << img.channels() << std::endl;
				   //std::cout << "num_channels_" << num_channels_ << std::endl;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;
	//图片尺寸处理
	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;//通道处理
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3, 1 / 255.0);
	else
		sample_resized.convertTo(sample_float, CV_32FC1, 1 / 255.0);
	cv::split(sample_float, *input_channels);/* 3通道数据分开存储 */
	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

//-----------------------------------------split()函数 -------------------------------------------------
/*************************************
data:2018.8.10
author:syy
function：SplitStrin（）：实现对输入字符去除指定的符号
parameter：1：输入string
2：输出vector
3.要去除的字符
version:1.0
思路：当去除指定字符后还需要添加空格于后面的字符便于区分，去除字符后加入vector会
导致vector以去除字符之间的字符串当做一个字符，导致后面vector的size（）变小，
所以还要以“ ”做为去除字符，使每个元素区分出来；

问题:1.shared_ptr<Net<float>> net_;shared_ptr报错，使用命名空间using namespace std;和caffe空间的冲突
把需要用到std命名空间的单独命名比如 using std::vector;

**************************************/
void SplitString(const std::string& intputString, std::vector<std::string>& outputString, const std::string& c)
{
	vector <string> outputTemp;
	string b = " ";
	std::string::size_type pos1, pos2;
	pos2 = intputString.find(c);
	pos1 = 0;
	string output;
	while (std::string::npos != pos2)//string::npos作为返回项表示未找到，（找到后循环）
	{
		outputTemp.push_back(intputString.substr(pos1, pos2 - pos1));//从pos1开始，复制pos2-pos1个元素
		outputTemp.push_back(" ");//去除标识后添加空格作为元素区分
		pos1 = pos2 + c.size();//从新设置起点，继续循环
		pos2 = intputString.find(c, pos1);//从新起点继续查找标识 
	}
	if (pos1 != intputString.length())//若后面没有需要的符号，把剩下数据拷
		outputTemp.push_back(intputString.substr(pos1));//

														//把中间vector值取出来赋值给string，用于除去“空格”
	for (auto it = outputTemp.begin(); it != outputTemp.end(); ++it)
		output += (*it);
	//cout <<"output"<<output;
	//////////////////
	pos2 = output.find(b);
	pos1 = 0;
	while (std::string::npos != pos2)//string::npos作为返回项表示未找到，（找到后循环）
	{
		outputString.push_back(output.substr(pos1, pos2 - pos1));//从pos1开始，复制pos2-pos1个元素
		pos1 = pos2 + b.size();//从新设置起点，继续循环
		pos2 = output.find(b, pos1);//从新起点继续查找标识 
	}
	if (pos1 != output.length())//若后面没有需要的符号，把剩下数据拷
		outputString.push_back(output.substr(pos1));//
}
//-----------------------------------------split()函数 -------------------------------

//-----------------------------------------numberConvert() --------------------------------------
//把string转成float 再转int型，直接转int会丢失信息报错
float StringConvertFloat(const std::string& intputString) {
	float Number;
	if (!(istringstream(intputString) >> Number)) {
		Number = 0;
		std::cout << intputString << "数据转化失败" << std::endl;
	}
	return Number;
}

string intConvertString(int intputInt)
{
	string outputString;
	ostringstream os; //构造一个输出字符串流，流内容为空 
	os << intputInt; //向输出字符串流中输出int整数i的内容 
	outputString = os.str(); //利用字符串流的str函数获取流中的内容 
	return outputString;
}

string floatConvertString(float intputInt)
{
	string outputString;
	ostringstream os; //构造一个输出字符串流，流内容为空 
	os << intputInt; //向输出字符串流中输出int整数i的内容 
	outputString = os.str(); //利用字符串流的str函数获取流中的内容 
	return outputString;
}
//-----------------------------------------numberConvert()  --------------------------------------

//-----------------------------------------列表文件读取裁剪 下 面-------------------------------------
/*************************************************
data:2018.8.10
author:syy
function：readList（）：实现对输入字符去除指定的符号
parameter：1：输入string
2：输出vector
3.要去除的字符
version:1.0
备注：为了与caffe框架风格保持一致，虽然使用了命名空间 namespace std，使用std namespace会与caffe的命名空间冲突

思路：1.当去除指定字符后还需要添加空格于后面的字符便于区分，去除字符后加入vector会导致vector以去除字符之间的字符串当做一个字符，
      导致后面vector的size（）变小，所以还要以“ ”做为去除字符，使每个元素区分出来；
	  2.vector容器在循环使用的时候要在第二次使用前清空clear（）；

*'数据格式：49142.jpg\t238 396 1256 1414\t512.643 676.214 851.929 1015.5 778.357 586.214 899.786 738.636 762.922 750.779 735.779 936.493 1122.21 1127.92
*需要去掉横向空格符号\t
*****************************************************/
void readTxt(string model_file, string trained_file, string testListFile, string imageFile, string save_file) {
	int lineNumber = 0;//当前读取的是第几行
	int lineSize = 19;
	int boundingWidth =0;
	int boundingHeigh = 0;
	string temp;
	string imageName;
	string outTxt;
	int boundingPositionXY[4];//0X1,1 Y1,2 X2,3Y2;
	int keyPointXY[14];//前7个是x，后7个是y
	vector <string> listLine;
	vector<float> getPredict;
	//----------------
	Regressioner regressioner(model_file, trained_file);
	//----------------读文件读取，统计行数------------------
	ifstream ReadFile;
	ReadFile.open(testListFile, ifstream::in); //需要加using namespace std，才能能使用ifstream
	if (ReadFile.fail()) {
		std::cout << "can not open testListFile" << std::endl;
	}
	else {
		std::cout << " open testListFile succefully" << std::endl;
		while (getline(ReadFile, temp)) {
			lineNumber++;
			SplitString(temp, listLine, "\t");//去除\t
			std::cout << "------>>>>当前行数：" << lineNumber << std::endl;
			if (listLine.size() != lineSize)
				std::cout << "------>>>>!!!!!!当前读取行成员个数不符合要求, " <<"当前读取行成员个数 " << listLine.size() << std::endl;
			std::cout << "------>>>>读取列表参数： ";
			for (auto it = listLine.begin(); it != listLine.end(); ++it) {
				std::cout << (*it) << " ";
			}
			//人脸框赋值
			imageName = listLine[0];
			for (int i = 0, j = 1; i < 4; i++, j++) {
				boundingPositionXY[i] = int(StringConvertFloat(listLine[j]));
				}
			//计算人脸框长宽
			boundingWidth = boundingPositionXY[3] - boundingPositionXY[1];//y2-y1
			boundingHeigh = boundingPositionXY[2] - boundingPositionXY[0];//x2-x1
			//人脸关键点赋值 
			for (int i = 0, j = 5; i < 14; i++, j++) {
				keyPointXY[i] = int(StringConvertFloat(listLine[j]));
			}

			
			string openImagePath = imageFile +imageName;
			std::cout <<"------>>>>:Open Image Path  "<< openImagePath <<std::endl;
			cv::Mat img_read = cv::imread(openImagePath,-1);//
			if(img_read.empty())
				std::cout << "------>>>>图像加载失败！"<<std::endl;
			//先行后列cv::Range的两个参数范围分别为左包含和右不包含
			cv::Mat img_crop = img_read(cv::Range(boundingPositionXY[1], boundingPositionXY[3] + 1),
										cv::Range(boundingPositionXY[0], boundingPositionXY[2] + 1));	
			cv::Mat img_gray;
			cv::cvtColor(img_crop, img_gray, CV_RGB2GRAY);
			getPredict = regressioner.Regression(img_gray);//预测关键点
			outTxt += imageName;
			outTxt += " ";
			std::cout << "预测值个数" << getPredict.size() << "预测值如下： ";
			for (int i =0; i<getPredict.size();i++) {
				outTxt += floatConvertString(getPredict[i]);
				outTxt += " ";
				std::cout << getPredict[i] <<" ";
				}
			outTxt += "\n";//换符号
			//标注出人脸框
			cv::rectangle(img_read, CvPoint(boundingPositionXY[0], boundingPositionXY[1]),
						CvPoint(boundingPositionXY[2], boundingPositionXY[3]), (255, 0, 0), 5);
		   //根据原来的标注画圈
			for (int i = 0; i < 7; i++) {
				//裁剪后的图片相对于原图发生了平移，所以标注数值也应该发生变化，减去左上坐标
				cv::circle(img_read,CvPoint(keyPointXY[i],keyPointXY[i + 7]), 5, cv::Scalar(0, 0, 255), -1);
				cv::circle(img_read,CvPoint(int(getPredict[i] * boundingWidth) + boundingPositionXY[0], int(getPredict[i+7]*boundingHeigh + boundingPositionXY[1])),
											5, cv::Scalar(0, 255, 0),-1);
				cv::putText(img_read, intConvertString(i), CvPoint(keyPointXY[i],keyPointXY[i + 7]),cv::FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 1);
			}

	
			cv::namedWindow("imageBoundingWindow", CV_WINDOW_AUTOSIZE);////在MyWindow的窗中中显示存储在img中的图片
			cv::imshow("imageBoundingWindow", img_read);
			cv::waitKey(0);////等待直到有键按下
			cv::destroyWindow("imageBoundingWindow");////销毁MyWindow的窗口
			
			string saveImagePath = save_file + imageName;
			string savePredictTxt = save_file + "imagePredict1.txt";
			std::ofstream outf;
			outf.open(savePredictTxt);//, ios::app
			outf << outTxt;
			outf.close();
			std::cout << "------>>>>:Save Image Path  " << saveImagePath << std::endl;
			cv::imwrite(saveImagePath, img_read);
			std::cout << imageName << "已经保存" << std::endl;
			//清空容器
			//outTxt.clear();
			openImagePath.clear();
			saveImagePath.clear();
			imageName.clear();//再次使用要求清空图片名字
			listLine.clear(); //再次读取前清vector空容器数据
			getPredict.clear();
		}
	}

	ReadFile.close();
	//return lineNumber;
}
//-----------------------------------------列表文件读取裁剪--------------------------------------------


int main(int argc, char** argv) {
	if (argc != 6) {
		std::cerr << "Usage: " << argv[0]
			<< " deploy.prototxt network.caffemodel"
			<< "testListFile  imageFile save_file" << std::endl;
		return 1;
	}

	::google::InitGoogleLogging(argv[0]);

	string model_file = argv[1];
	string trained_file = argv[2];
	string testListFile = argv[3];
	string imageFile = argv[4];
	string save_file = argv[5];
	readTxt(model_file, trained_file, testListFile, imageFile, save_file);
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV