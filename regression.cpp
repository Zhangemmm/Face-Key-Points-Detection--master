//#include "caffe/caffe.hpp"
//
//int main(int argc, char** argv) {
//  LOG(FATAL) << "Deprecated. Use caffe train --solver=... "
//                "[--snapshot=...] instead.";
//  return 0;
//}

/**********************************************************
�ع�Ԥ�⣺
���˼·��
1.����಻ͬ�ľ���������ͬ
2.��Ҫ��ֵ�ļ�������ͼƬ��һ������
���ݷ�����޸�
���룺deploynetģ�ͣ�caffemodel����ֵ����Ԥ��ͼƬ
2.�������̣�
2.1ָ�����㵥Ԫ----��2.2caffe����net��������ģ�ͣ�����ѵ���׶Σ�
----��2.3net-blob��ȡ���ݲ�reshape---��2.4��ȡ��ֵ����ȡֵ��Χ
----��2.5Ԥ����ͼƬ------��2.6blob���ش�����ͼƬ
------��2.7����net_forward()��ȡoutputֵ

ʵ�ʣ�regression��������������磬predictic��WrapInputLayer��Preprocess����ֵ����net_->Forward();��
**************************************************/

//#include <caffe/caffe.hpp>
//#ifdef USE_OPENCV
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#endif  // USE_OPENCV
//#include <algorithm>
//#include <iosfwd>
//#include <memory>
//#include <string>
//#include <utility>
//#include <vector>
//
//#include <iomanip>//���������С����λ��<< setprecision(3)
//#ifdef USE_OPENCV
//using namespace caffe;  // NOLINT(build/namespaces)
//using std::string;
//
///* Pair (label, confidence) representing a prediction. */
//typedef std::pair<string, float> Prediction;
////�������
//class Regressioner {
//public:
//	Regressioner(const string& model_file, //���캯������������ģ�ͣ�ѵ���õ�model
//		const string& trained_file);//const string& mean_file ȥ����ֵ����
//	std::vector<Prediction> Regression(const cv::Mat& img);	//Regressio,�������ͼ��
//private:
//	//���ξ�ֵ����
//	//void SetMean(const string& mean_file); //setMean�Ƕ����ֵ�ļ���ת����һ�ž�ֵͼ���mean_,�β��Ǿ�ֵ�ļ����ļ���
//	std::vector<float> Predict(const cv::Mat& img);//predict��������process������ͼƬ���������У�ʹ��net_->Forward()��������Ԥ��
//												   //������� ������浽vector�����з��أ������β��ǵ���ͼƬ
//	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
//	void Preprocess(const cv::Mat& img,      //preprocess������ͼ��ͨ������С��������ʽ���иı䣬��ȥ��ֵ����д��net��������
//		std::vector<cv::Mat>* input_channels);
//private://���˽�б���
//	shared_ptr<Net<float>> net_;//����
//	cv::Size input_geometry_;//�����ͼ���С
//	int num_channels_;//����ͨ����
//	//���ξ�ֵ�ͱ�ǩ
//	//cv::Mat mean_;//��ֵ
//	//std::vector<string> labels_;//��ǩ
//};
//
////�����ڵ������������//const string& mean_file
//
//Regressioner::Regressioner(const string& model_file,
//	const string& trained_file) {
//#ifdef CPU_ONLY
//	Caffe::set_mode(Caffe::CPU);
//#else
//	Caffe::set_mode(Caffe::GPU);
//#endif
//
//	/* Load the network. */
//	net_.reset(new Net<float>(model_file, TEST));//���������ļ� ��ģʽΪtest
//	net_->CopyTrainedLayersFrom(trained_file);//����caffemodel���ú�����net.cpp��ʵ��  //Ҫ�������������1ָ����blob����
//	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
//	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
//
//	Blob<float>* input_layer = net_->input_blobs()[0];
//	num_channels_ = input_layer->channels();
//	CHECK(num_channels_ == 3 || num_channels_ == 1)//�ж�ͨ����
//		<< "Input layer should have 1 or 3 channels.";
//	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
//
//	//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&ȥ�������ֵ�Ĳ���
//	/* Load the binaryproto mean file. */
//	//SetMean(mean_file);
//
//	Blob<float>* output_layer = net_->output_blobs()[0];// �����ֻ��һ��Blob�������[0]; ���⣬������shapeΪ��1, 10��
//}
//
//std::vector<Prediction> Regressioner::Regression(const cv::Mat& img) {
//
//
//	std::vector<float> output;
//	output = Predict(img);// ����Predict����������ͼ�����Ԥ��
//	std::cout << "Ԥ�����������" << output.size() << std::endl;
//	for (int i = 0; i<output.size(); i++) {
//		std::cout << output[i] << " ";
//	}
//	//std::cout <<"\n"<< std::endl;
//}
///* Load the mean file in binaryproto format. */
////���ؾ�ֵ�����Ķ���
////void Regressioner::SetMean(const string& mean_file) {
////	BlobProto blob_proto;
////	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);//��ȡ��ֵ�ļ���blob
////
////																 /* Convert from BlobProto to Blob<float> */
////	Blob<float> mean_blob;//ת�ɸ�����
////	mean_blob.FromProto(blob_proto);//������mean_blob
////	CHECK_EQ(mean_blob.channels(), num_channels_)
////		<< "Number of channels of mean file doesn't match input layer.";
////
////	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
////	// ����ͨ����ͼƬ�ֿ��洢������ͼƬBGR��˳�򱣴浽channels�� (����mnist,ֻ��һ��ͨ���������������ͨ�õķ���)
////	std::vector<cv::Mat> channels;
////	float* data = mean_blob.mutable_cpu_data();
////	for (int i = 0; i < num_channels_; ++i) {
////		/* Extract an individual channel. */
////		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
////		channels.push_back(channel);
////		data += mean_blob.height() * mean_blob.width();
////	}
////
////	/* Merge the separate channels into a single image. */
////	// ���ºϳ�һ��ͼƬ
////	cv::Mat mean;
////	cv::merge(channels, mean);
////	// ����ÿ��ͨ���ľ�ֵ���õ�һ����ά������channel_mean��Ȼ�����ά��������չ��һ���µľ�ֵͼƬ  
////	// ����ͼƬ��ÿ��ͨ��������ֵ����ȵģ����ž�ֵͼƬ�Ĵ�С�������������Ҫ��һ�� 
////	// ע�⣺ �����ȥ��ֵ����ָ����Ҫ�����ͼ���ȥ��ֵͼ���ƽ������ 
////	/* Compute the global mean pixel value and create a mean image
////	* filled with this value. */
////	cv::Scalar channel_mean = cv::mean(mean);
////	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
////}
//
//std::vector<float> Regressioner::Predict(const cv::Mat& img) {  //�����β�Ϊ����ͼ�� 
//	Blob<float>* input_layer = net_->input_blobs()[0];//���������
//	input_layer->Reshape(1, num_channels_,input_geometry_.height, input_geometry_.width);//��������ͼƬ�ߴ�
//	/* Forward dimension change to all layers. */
//	net_->Reshape();
//	std::vector<cv::Mat> input_channels;
//	/* Ϊ�˻��net_�������������ݵ�ָ�룬ֱ�Ӱ�����ͼƬ���ݿ��������ָ������*/// ��cv::Mat����ͼ�����ݵ�size��channel�Ⱥ���·������Blobg����������
//	WrapInputLayer(&input_channels);
//	// �����Ԥ���ͼƬ���ݣ�Ȼ�����Ԥ����������һ�������ŵȲ���  
//	Preprocess(img, &input_channels);
//	net_->Forward();//ǰ���
//	/* Copy the output layer to a std::vector */
//	Blob<float>* output_layer = net_->output_blobs()[0];//��������
//	const float* begin = output_layer->cpu_data();
//	const float* end = begin + output_layer->channels();
//	return std::vector<float>(begin, end);
//}
//void Regressioner::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
//	Blob<float>* input_layer = net_->input_blobs()[0];//��������
//
//	int width = input_layer->width();
//	int height = input_layer->height();
//	float* input_data = input_layer->mutable_cpu_data();
//	for (int i = 0; i < input_layer->channels(); ++i) {
//		cv::Mat channel(height, width, CV_32FC1, input_data);
//		input_channels->push_back(channel);
//		input_data += width * height;
//	}
//}
//// ͼƬԤ������������ͼƬ���š���һ����3ͨ��ͼƬ�ֿ��洢
//void Regressioner::Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels) {
//	/* Convert the input image to the input image format of the network. */
//	cv::Mat sample;//����ͨ������ͼƬ��ɫ
//	if (img.channels() == 3 && num_channels_ == 1)
//		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
//	else if (img.channels() == 4 && num_channels_ == 1)
//		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
//	else if (img.channels() == 4 && num_channels_ == 3)
//		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
//	else if (img.channels() == 1 && num_channels_ == 3)
//		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
//	else
//
//		sample = img;
//	//ͼƬ�ߴ紦��
//	cv::Mat sample_resized;
//	if (sample.size() != input_geometry_)
//		cv::resize(sample, sample_resized, input_geometry_);
//	else
//		sample_resized = sample;
//    //ͨ������
//	cv::Mat sample_float;
//	if (num_channels_ == 3)
//		//@@@@@@@@@@@@@@@@@@@@@@@@@���Ӳ���@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//		//sample_resized.convertTo(sample_float, CV_32FC3);
//		sample_resized.convertTo(sample_float, CV_32FC3, 1/255.0);
//	else
//		//sample_resized.convertTo(sample_float, CV_32FC1);
//		sample_resized.convertTo(sample_float, CV_32FC3, 1/255.0);
//	
//	//��ֵ��һ����
//	
//	//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&û�м����ֵȥ��
//	 //cv::Mat sample_normalized;//ͼ�����
//	 //cv::subtract(sample_float, mean_, sample_normalized);
//
//	/* This operation will write the separate BGR planes directly to the
//	* input layer of the network because it is wrapped by the cv::Mat
//	* objects in input_channels. */
//	  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//	  //cv::split(sample_normalized, *input_channels);/* 3ͨ�����ݷֿ��洢 */
//	//@@@@@@@@@@@@@@@@@@@@@@@@@���Ӳ���@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//	cv::split(sample_float, *input_channels);
//	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
//		== net_->input_blobs()[0]->cpu_data())
//		<< "Input channels are not wrapping the input layer of the network.";
//}
//
//int main(int argc, char** argv) {
//	if (argc != 4) {
//		std::cerr << "Usage: " << argv[0]
//			<< " deploy.prototxt network.caffemodel"
//			<< " img.jpg" << std::endl;
//		return 1;
//	}
//
//	::google::InitGoogleLogging(argv[0]);
//
//	string model_file = argv[1];
//	string trained_file = argv[2];
//	//string mean_file = argv[3];
//	Regressioner regressioner(model_file, trained_file);//, mean_file
//
//	string file = argv[3];
//
//	std::cout << "---------- Prediction for "
//		<< file << " ----------" << std::endl;
//
//	cv::Mat img = cv::imread(file, -1);
//	CHECK(!img.empty()) << "Unable to decode image " << file;
//	std::vector<Prediction> predictions = regressioner.Regression(img);
//
//}
//#else
//int main(int argc, char** argv) {
//	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
//}
//#endif  // USE_OPENCV
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////����汾/////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

#include <iomanip>//���������С����λ��<< setprecision(3)
#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;
class Regressioner {
public:
	Regressioner(const string& model_file, //���캯������������ģ�ͣ�ѵ���õ�model
		const string& trained_file);//const string& mean_file ȥ����ֵ����
	std::vector<Prediction> Regression(const cv::Mat& img);	//Regressio,�������ͼ��
private:
	std::vector<float> Predict(const cv::Mat& img);//predict��������process������ͼƬ���������У�ʹ��net_->Forward()��������Ԥ��											 
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);
private://���˽�б���
	shared_ptr<Net<float>> net_;//����
	cv::Size input_geometry_;//�����ͼ���С
	int num_channels_;//����ͨ����
};

Regressioner::Regressioner(const string& model_file,const string& trained_file) {
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

std::vector<Prediction> Regressioner::Regression(const cv::Mat& img) {
	std::vector<float> output;
	output = Predict(img);// ����Predict����������ͼ�����Ԥ��
	std::cout << "Ԥ�����������" << output.size() << std::endl;
	for (int i = 0; i<output.size(); i++) {
		std::cout << output[i] << " ";
	}
	//std::cout <<"\n"<< std::endl;
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

int main(int argc, char** argv) {
	if (argc != 4) {
		std::cerr << "Usage: " << argv[0]
			<< " deploy.prototxt network.caffemodel"
			<< " img.jpg" << std::endl;
		return 1;
	}

	::google::InitGoogleLogging(argv[0]);

	string model_file = argv[1];
	string trained_file = argv[2];
	//string mean_file = argv[3];
	Regressioner regressioner(model_file, trained_file);//, mean_file

	string file = argv[3];

	std::cout << "---------- Prediction for "
		<< file << " ----------" << std::endl;

	cv::Mat img = cv::imread(file, -1);
	CHECK(!img.empty()) << "Unable to decode image " << file;
	std::vector<Prediction> predictions = regressioner.Regression(img);

}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV