//#include "caffe/caffe.hpp"
//
//int main(int argc, char** argv) {
//  LOG(FATAL) << "Deprecated. Use caffe train --solver=... "
//                "[--snapshot=...] instead.";
//  return 0;
//}

/**********************************************************
回归预测：
编程思路：
1.与分类不同的就是损函数不同
2.不要均值文件，采用图片归一化处理
根据分类的修改
输入：deploynet模型，caffemodel，均值，被预测图片
2.程序流程：
2.1指定计算单元----》2.2caffe——net加载网络模型（包括训练阶段）
----》2.3net-blob读取数据并reshape---》2.4读取均值，和取值范围
----》2.5预处理图片------》2.6blob加载处理后的图片
------》2.7运行net_forward()获取output值

实际：regression（运行类加载网络，predictic（WrapInputLayer，Preprocess（均值），net_->Forward();）
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
//#include <iomanip>//控制输出的小数点位数<< setprecision(3)
//#ifdef USE_OPENCV
//using namespace caffe;  // NOLINT(build/namespaces)
//using std::string;
//
///* Pair (label, confidence) representing a prediction. */
//typedef std::pair<string, float> Prediction;
////类的声明
//class Regressioner {
//public:
//	Regressioner(const string& model_file, //构造函数声明，网络模型，训练好的model
//		const string& trained_file);//const string& mean_file 去掉均值输入
//	std::vector<Prediction> Regression(const cv::Mat& img);	//Regressio,对输入的图形
//private:
//	//屏蔽均值计算
//	//void SetMean(const string& mean_file); //setMean是读入均值文件，转化成一张均值图像的mean_,形参是均值文件的文件名
//	std::vector<float> Predict(const cv::Mat& img);//predict函数调用process函数将图片输入网络中，使用net_->Forward()函数进行预测
//												   //将输出层 输出保存到vector容器中返回，输入形参是单张图片
//	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
//	void Preprocess(const cv::Mat& img,      //preprocess函数对图像通道，大小，数据形式进行改变，减去均值，再写到net的输入中
//		std::vector<cv::Mat>* input_channels);
//private://类的私有变量
//	shared_ptr<Net<float>> net_;//网络
//	cv::Size input_geometry_;//输入层图像大小
//	int num_channels_;//输入通道数
//	//屏蔽均值和标签
//	//cv::Mat mean_;//均值
//	//std::vector<string> labels_;//标签
//};
//
////在类内的网络参数加载//const string& mean_file
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
//	net_.reset(new Net<float>(model_file, TEST));//加载配置文件 ，模式为test
//	net_->CopyTrainedLayersFrom(trained_file);//加载caffemodel，该函数再net.cpp中实现  //要求输入输出都是1指的是blob个数
//	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
//	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
//
//	Blob<float>* input_layer = net_->input_blobs()[0];
//	num_channels_ = input_layer->channels();
//	CHECK(num_channels_ == 3 || num_channels_ == 1)//判断通道数
//		<< "Input layer should have 1 or 3 channels.";
//	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
//
//	//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&去掉计算均值的部分
//	/* Load the binaryproto mean file. */
//	//SetMean(mean_file);
//
//	Blob<float>* output_layer = net_->output_blobs()[0];// 输出层只有一个Blob，因此用[0]; 另外，输出层的shape为（1, 10）
//}
//
//std::vector<Prediction> Regressioner::Regression(const cv::Mat& img) {
//
//
//	std::vector<float> output;
//	output = Predict(img);// 调用Predict函数对输入图像进行预测
//	std::cout << "预测输出个数：" << output.size() << std::endl;
//	for (int i = 0; i<output.size(); i++) {
//		std::cout << output[i] << " ";
//	}
//	//std::cout <<"\n"<< std::endl;
//}
///* Load the mean file in binaryproto format. */
////加载均值函数的定义
////void Regressioner::SetMean(const string& mean_file) {
////	BlobProto blob_proto;
////	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);//读取均值文件给blob
////
////																 /* Convert from BlobProto to Blob<float> */
////	Blob<float> mean_blob;//转成浮点型
////	mean_blob.FromProto(blob_proto);//拷贝给mean_blob
////	CHECK_EQ(mean_blob.channels(), num_channels_)
////		<< "Number of channels of mean file doesn't match input layer.";
////
////	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
////	// 把三通道的图片分开存储，三张图片BGR按顺序保存到channels中 (对于mnist,只有一个通道；这里给出的是通用的方法)
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
////	// 重新合成一张图片
////	cv::Mat mean;
////	cv::merge(channels, mean);
////	// 计算每个通道的均值，得到一个三维的向量channel_mean，然后把三维的向量扩展成一张新的均值图片  
////	// 这种图片的每个通道的像素值是相等的，这张均值图片的大小将和网络的输入要求一样 
////	// 注意： 这里的去均值，是指对需要处理的图像减去均值图像的平均亮度 
////	/* Compute the global mean pixel value and create a mean image
////	* filled with this value. */
////	cv::Scalar channel_mean = cv::mean(mean);
////	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
////}
//
//std::vector<float> Regressioner::Predict(const cv::Mat& img) {  //输入形参为单张图像 
//	Blob<float>* input_layer = net_->input_blobs()[0];//网络输入层
//	input_layer->Reshape(1, num_channels_,input_geometry_.height, input_geometry_.width);//调整输入图片尺寸
//	/* Forward dimension change to all layers. */
//	net_->Reshape();
//	std::vector<cv::Mat> input_channels;
//	/* 为了获得net_网络的输入层数据的指针，直接把输入图片数据拷贝到这个指针里面*/// 将cv::Mat类型图像数据的size、channel等和网路输入层的Blobg关联起来。
//	WrapInputLayer(&input_channels);
//	// 输入带预测的图片数据，然后进行预处理，包括归一化、缩放等操作  
//	Preprocess(img, &input_channels);
//	net_->Forward();//前项传播
//	/* Copy the output layer to a std::vector */
//	Blob<float>* output_layer = net_->output_blobs()[0];//网络层输出
//	const float* begin = output_layer->cpu_data();
//	const float* end = begin + output_layer->channels();
//	return std::vector<float>(begin, end);
//}
//void Regressioner::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
//	Blob<float>* input_layer = net_->input_blobs()[0];//网络输入
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
//// 图片预处理函数，包括图片缩放、归一化、3通道图片分开存储
//void Regressioner::Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels) {
//	/* Convert the input image to the input image format of the network. */
//	cv::Mat sample;//根据通道设置图片颜色
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
//	//图片尺寸处理
//	cv::Mat sample_resized;
//	if (sample.size() != input_geometry_)
//		cv::resize(sample, sample_resized, input_geometry_);
//	else
//		sample_resized = sample;
//    //通道处理
//	cv::Mat sample_float;
//	if (num_channels_ == 3)
//		//@@@@@@@@@@@@@@@@@@@@@@@@@增加部分@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//		//sample_resized.convertTo(sample_float, CV_32FC3);
//		sample_resized.convertTo(sample_float, CV_32FC3, 1/255.0);
//	else
//		//sample_resized.convertTo(sample_float, CV_32FC1);
//		sample_resized.convertTo(sample_float, CV_32FC3, 1/255.0);
//	
//	//均值归一化，
//	
//	//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&没有计算均值去掉
//	 //cv::Mat sample_normalized;//图像相减
//	 //cv::subtract(sample_float, mean_, sample_normalized);
//
//	/* This operation will write the separate BGR planes directly to the
//	* input layer of the network because it is wrapped by the cv::Mat
//	* objects in input_channels. */
//	  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//	  //cv::split(sample_normalized, *input_channels);/* 3通道数据分开存储 */
//	//@@@@@@@@@@@@@@@@@@@@@@@@@增加部分@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
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
//////////////////////////////////精简版本/////////////////////////////////////////////////////////////////
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

#include <iomanip>//控制输出的小数点位数<< setprecision(3)
#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;
class Regressioner {
public:
	Regressioner(const string& model_file, //构造函数声明，网络模型，训练好的model
		const string& trained_file);//const string& mean_file 去掉均值输入
	std::vector<Prediction> Regression(const cv::Mat& img);	//Regressio,对输入的图形
private:
	std::vector<float> Predict(const cv::Mat& img);//predict函数调用process函数将图片输入网络中，使用net_->Forward()函数进行预测											 
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);
private://类的私有变量
	shared_ptr<Net<float>> net_;//网络
	cv::Size input_geometry_;//输入层图像大小
	int num_channels_;//输入通道数
};

Regressioner::Regressioner(const string& model_file,const string& trained_file) {
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

std::vector<Prediction> Regressioner::Regression(const cv::Mat& img) {
	std::vector<float> output;
	output = Predict(img);// 调用Predict函数对输入图像进行预测
	std::cout << "预测输出个数：" << output.size() << std::endl;
	for (int i = 0; i<output.size(); i++) {
		std::cout << output[i] << " ";
	}
	//std::cout <<"\n"<< std::endl;
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