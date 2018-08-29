///////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// ��ǿ��/////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
/***********************************************
*Date:2018.8.9
*author=syy
*�������ܣ������ؼ���ع�Ԥ��
*���������1.����ģ��Deploy 2.ѵ���õ�����ģ��caffemodel 3.��֤��list��ַ
*         4.��֤ͼƬ��ַ 5.Ԥ��ֵ�����ַ�������ļ���) 6.��ע�õ�ͼƬ�����ַ
*�����1.��֤��Ԥ��ֵ�ļ� 2.��ע�õ�ͼƬ
*Version��v1.3 ��1.�����ע�ļ���ͼƬֱ�Ӹ���Ԥ��ֵ�ͱ�ע�õ�ͼƬ
2.ʵ����������ͼƬ��Ԥ��
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
#include <sstream>//stringת��floatͷ�ļ�
//#include <iomanip>//���������С����λ��<< setprecision(3)
#ifdef USE_OPENCV
using namespace caffe;// NOLINT(build/namespaces)
//������caffe�����ռ��ͻ
using std::string;
using std::vector;
using std::ifstream;
using std::istringstream;
/* Pair (label, confidence) representing a prediction. */
class Regressioner {
public:
	Regressioner(const string& model_file, //���캯������������ģ�ͣ�ѵ���õ�model
		const string& trained_file);//const string& mean_file ȥ����ֵ����
	std::vector<float> Regression(const cv::Mat& img);	//Regressio,�������ͼ��
private:
	std::vector<float> Predict(const cv::Mat& img);//predict��������process������ͼƬ���������У�ʹ��net_->Forward()��������Ԥ��											 
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
private://���˽�б���
	shared_ptr<Net<float>> net_;//����
	cv::Size input_geometry_;//�����ͼ���С
	int num_channels_;//����ͨ����
};

Regressioner::Regressioner(const string& model_file, const string& trained_file) {
	Caffe::set_mode(Caffe::GPU);
	net_.reset(new Net<float>(model_file, TEST));//���������ļ� ��ģʽΪtest
	net_->CopyTrainedLayersFrom(trained_file);//����caffemodel���ú�����net.cpp��ʵ��  //Ҫ�������������1ָ����blob����
	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)//�ж�ͨ����
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	Blob<float>* output_layer = net_->output_blobs()[0];// �����ֻ��һ��Blob�������[0]; ���⣬������shapeΪ��1, 10��
}

std::vector<float> Regressioner::Regression(const cv::Mat& img) {
	std::vector<float> output = Predict(img);// ����Predict����������ͼ�����Ԥ��
	return output;
}

std::vector<float> Regressioner::Predict(const cv::Mat& img) {  //�����β�Ϊ����ͼ�� 
	Blob<float>* input_layer = net_->input_blobs()[0];//���������
	input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);//��������ͼƬ�ߴ�
																						  /* Forward dimension change to all layers. */
	net_->Reshape();
	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);	// Ϊ�˻��net_�������������ݵ�ָ�룬ֱ�Ӱ�����ͼƬ���ݿ��������ָ������
	Preprocess(img, &input_channels);	// �����Ԥ���ͼƬ���ݣ�Ȼ�����Ԥ����������һ�������ŵȲ���  
	net_->Forward();//ǰ���
	Blob<float>* output_layer = net_->output_blobs()[0];/* Copy the output layer to a std::vector */
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}
void Regressioner::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];//��������

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Regressioner::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {// ͼƬԤ������������ͼƬ���š���һ����3ͨ��ͼƬ�ֿ��洢
																						 /* Convert the input image to the input image format of the network. */
	cv::Mat sample;//����ͨ������ͼƬ��ɫ
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
	//ͼƬ�ߴ紦��
	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;//ͨ������
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3, 1 / 255.0);
	else
		sample_resized.convertTo(sample_float, CV_32FC1, 1 / 255.0);
	cv::split(sample_float, *input_channels);/* 3ͨ�����ݷֿ��洢 */
	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

//-----------------------------------------split()���� -------------------------------------------------
/*************************************
data:2018.8.10
author:syy
function��SplitStrin������ʵ�ֶ������ַ�ȥ��ָ���ķ���
parameter��1������string
2�����vector
3.Ҫȥ�����ַ�
version:1.0
˼·����ȥ��ָ���ַ�����Ҫ��ӿո��ں�����ַ��������֣�ȥ���ַ������vector��
����vector��ȥ���ַ�֮����ַ�������һ���ַ������º���vector��size������С��
���Ի�Ҫ�ԡ� ����Ϊȥ���ַ���ʹÿ��Ԫ�����ֳ�����

����:1.shared_ptr<Net<float>> net_;shared_ptr����ʹ�������ռ�using namespace std;��caffe�ռ�ĳ�ͻ
����Ҫ�õ�std�����ռ�ĵ����������� using std::vector;

**************************************/
void SplitString(const std::string& intputString, std::vector<std::string>& outputString, const std::string& c)
{
	vector <string> outputTemp;
	string b = " ";
	std::string::size_type pos1, pos2;
	pos2 = intputString.find(c);
	pos1 = 0;
	string output;
	while (std::string::npos != pos2)//string::npos��Ϊ�������ʾδ�ҵ������ҵ���ѭ����
	{
		outputTemp.push_back(intputString.substr(pos1, pos2 - pos1));//��pos1��ʼ������pos2-pos1��Ԫ��
		outputTemp.push_back(" ");//ȥ����ʶ����ӿո���ΪԪ������
		pos1 = pos2 + c.size();//����������㣬����ѭ��
		pos2 = intputString.find(c, pos1);//�������������ұ�ʶ 
	}
	if (pos1 != intputString.length())//������û����Ҫ�ķ��ţ���ʣ�����ݿ�
		outputTemp.push_back(intputString.substr(pos1));//

														//���м�vectorֵȡ������ֵ��string�����ڳ�ȥ���ո�
	for (auto it = outputTemp.begin(); it != outputTemp.end(); ++it)
		output += (*it);
	//cout <<"output"<<output;
	//////////////////
	pos2 = output.find(b);
	pos1 = 0;
	while (std::string::npos != pos2)//string::npos��Ϊ�������ʾδ�ҵ������ҵ���ѭ����
	{
		outputString.push_back(output.substr(pos1, pos2 - pos1));//��pos1��ʼ������pos2-pos1��Ԫ��
		pos1 = pos2 + b.size();//����������㣬����ѭ��
		pos2 = output.find(b, pos1);//�������������ұ�ʶ 
	}
	if (pos1 != output.length())//������û����Ҫ�ķ��ţ���ʣ�����ݿ�
		outputString.push_back(output.substr(pos1));//
}
//-----------------------------------------split()���� -------------------------------

//-----------------------------------------numberConvert() --------------------------------------
//��stringת��float ��תint�ͣ�ֱ��תint�ᶪʧ��Ϣ����
float StringConvertFloat(const std::string& intputString) {
	float Number;
	if (!(istringstream(intputString) >> Number)) {
		Number = 0;
		std::cout << intputString << "����ת��ʧ��" << std::endl;
	}
	return Number;
}

string intConvertString(int intputInt)
{
	string outputString;
	ostringstream os; //����һ������ַ�������������Ϊ�� 
	os << intputInt; //������ַ����������int����i������ 
	outputString = os.str(); //�����ַ�������str������ȡ���е����� 
	return outputString;
}

string floatConvertString(float intputInt)
{
	string outputString;
	ostringstream os; //����һ������ַ�������������Ϊ�� 
	os << intputInt; //������ַ����������int����i������ 
	outputString = os.str(); //�����ַ�������str������ȡ���е����� 
	return outputString;
}
//-----------------------------------------numberConvert()  --------------------------------------

//-----------------------------------------�б��ļ���ȡ�ü� �� ��-------------------------------------
/*************************************************
data:2018.8.10
author:syy
function��readList������ʵ�ֶ������ַ�ȥ��ָ���ķ���
parameter��1������string
2�����vector
3.Ҫȥ�����ַ�
version:1.0
��ע��Ϊ����caffe��ܷ�񱣳�һ�£���Ȼʹ���������ռ� namespace std��ʹ��std namespace����caffe�������ռ��ͻ

˼·��1.��ȥ��ָ���ַ�����Ҫ��ӿո��ں�����ַ��������֣�ȥ���ַ������vector�ᵼ��vector��ȥ���ַ�֮����ַ�������һ���ַ���
      ���º���vector��size������С�����Ի�Ҫ�ԡ� ����Ϊȥ���ַ���ʹÿ��Ԫ�����ֳ�����
	  2.vector������ѭ��ʹ�õ�ʱ��Ҫ�ڵڶ���ʹ��ǰ���clear������

*'���ݸ�ʽ��49142.jpg\t238 396 1256 1414\t512.643 676.214 851.929 1015.5 778.357 586.214 899.786 738.636 762.922 750.779 735.779 936.493 1122.21 1127.92
*��Ҫȥ������ո����\t
*****************************************************/
void readTxt(string model_file, string trained_file, string testListFile, string imageFile, string save_file) {
	int lineNumber = 0;//��ǰ��ȡ���ǵڼ���
	int lineSize = 19;
	int boundingWidth =0;
	int boundingHeigh = 0;
	string temp;
	string imageName;
	string outTxt;
	int boundingPositionXY[4];//0X1,1 Y1,2 X2,3Y2;
	int keyPointXY[14];//ǰ7����x����7����y
	vector <string> listLine;
	vector<float> getPredict;
	//----------------
	Regressioner regressioner(model_file, trained_file);
	//----------------���ļ���ȡ��ͳ������------------------
	ifstream ReadFile;
	ReadFile.open(testListFile, ifstream::in); //��Ҫ��using namespace std��������ʹ��ifstream
	if (ReadFile.fail()) {
		std::cout << "can not open testListFile" << std::endl;
	}
	else {
		std::cout << " open testListFile succefully" << std::endl;
		while (getline(ReadFile, temp)) {
			lineNumber++;
			SplitString(temp, listLine, "\t");//ȥ��\t
			std::cout << "------>>>>��ǰ������" << lineNumber << std::endl;
			if (listLine.size() != lineSize)
				std::cout << "------>>>>!!!!!!��ǰ��ȡ�г�Ա����������Ҫ��, " <<"��ǰ��ȡ�г�Ա���� " << listLine.size() << std::endl;
			std::cout << "------>>>>��ȡ�б������ ";
			for (auto it = listLine.begin(); it != listLine.end(); ++it) {
				std::cout << (*it) << " ";
			}
			//������ֵ
			imageName = listLine[0];
			for (int i = 0, j = 1; i < 4; i++, j++) {
				boundingPositionXY[i] = int(StringConvertFloat(listLine[j]));
				}
			//���������򳤿�
			boundingWidth = boundingPositionXY[3] - boundingPositionXY[1];//y2-y1
			boundingHeigh = boundingPositionXY[2] - boundingPositionXY[0];//x2-x1
			//�����ؼ��㸳ֵ 
			for (int i = 0, j = 5; i < 14; i++, j++) {
				keyPointXY[i] = int(StringConvertFloat(listLine[j]));
			}

			
			string openImagePath = imageFile +imageName;
			std::cout <<"------>>>>:Open Image Path  "<< openImagePath <<std::endl;
			cv::Mat img_read = cv::imread(openImagePath,-1);//
			if(img_read.empty())
				std::cout << "------>>>>ͼ�����ʧ�ܣ�"<<std::endl;
			//���к���cv::Range������������Χ�ֱ�Ϊ��������Ҳ�����
			cv::Mat img_crop = img_read(cv::Range(boundingPositionXY[1], boundingPositionXY[3] + 1),
										cv::Range(boundingPositionXY[0], boundingPositionXY[2] + 1));	
			cv::Mat img_gray;
			cv::cvtColor(img_crop, img_gray, CV_RGB2GRAY);
			getPredict = regressioner.Regression(img_gray);//Ԥ��ؼ���
			outTxt += imageName;
			outTxt += " ";
			std::cout << "Ԥ��ֵ����" << getPredict.size() << "Ԥ��ֵ���£� ";
			for (int i =0; i<getPredict.size();i++) {
				outTxt += floatConvertString(getPredict[i]);
				outTxt += " ";
				std::cout << getPredict[i] <<" ";
				}
			outTxt += "\n";//������
			//��ע��������
			cv::rectangle(img_read, CvPoint(boundingPositionXY[0], boundingPositionXY[1]),
						CvPoint(boundingPositionXY[2], boundingPositionXY[3]), (255, 0, 0), 5);
		   //����ԭ���ı�ע��Ȧ
			for (int i = 0; i < 7; i++) {
				//�ü����ͼƬ�����ԭͼ������ƽ�ƣ����Ա�ע��ֵҲӦ�÷����仯����ȥ��������
				cv::circle(img_read,CvPoint(keyPointXY[i],keyPointXY[i + 7]), 5, cv::Scalar(0, 0, 255), -1);
				cv::circle(img_read,CvPoint(int(getPredict[i] * boundingWidth) + boundingPositionXY[0], int(getPredict[i+7]*boundingHeigh + boundingPositionXY[1])),
											5, cv::Scalar(0, 255, 0),-1);
				cv::putText(img_read, intConvertString(i), CvPoint(keyPointXY[i],keyPointXY[i + 7]),cv::FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 1);
			}

	
			cv::namedWindow("imageBoundingWindow", CV_WINDOW_AUTOSIZE);////��MyWindow�Ĵ�������ʾ�洢��img�е�ͼƬ
			cv::imshow("imageBoundingWindow", img_read);
			cv::waitKey(0);////�ȴ�ֱ���м�����
			cv::destroyWindow("imageBoundingWindow");////����MyWindow�Ĵ���
			
			string saveImagePath = save_file + imageName;
			string savePredictTxt = save_file + "imagePredict1.txt";
			std::ofstream outf;
			outf.open(savePredictTxt);//, ios::app
			outf << outTxt;
			outf.close();
			std::cout << "------>>>>:Save Image Path  " << saveImagePath << std::endl;
			cv::imwrite(saveImagePath, img_read);
			std::cout << imageName << "�Ѿ�����" << std::endl;
			//�������
			//outTxt.clear();
			openImagePath.clear();
			saveImagePath.clear();
			imageName.clear();//�ٴ�ʹ��Ҫ�����ͼƬ����
			listLine.clear(); //�ٴζ�ȡǰ��vector����������
			getPredict.clear();
		}
	}

	ReadFile.close();
	//return lineNumber;
}
//-----------------------------------------�б��ļ���ȡ�ü�--------------------------------------------


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