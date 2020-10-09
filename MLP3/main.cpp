#include "network.h"
 
#include "fstream"
#include "iostream" 
#include <string>
#include <sstream>

using namespace mlp;  
using namespace std;

//#define GENERATE_WEIGHT 
//#define GENERATE_FAULT
//#define GENERATE_FAULT_TYPE
#define RETRAIN  
//#define REMAP 

int main(int argc,char* argv[]){
	vec2d_t train_x;
	vec_t train_y;
	vec2d_t test_x;
	vec_t test_y;
	//std::ofstream ofile("weight.txt");
	/*加载MNIST数据集*/
	LOAD_MNIST_TEST(test_x, test_y);
	LOAD_MNIST_TRAIN(train_x, train_y);
	
	/*使用神经网络拟合XOR函数，快速验证神经网络正确性*/
	//vec2d_t XOR_x = { { 0, 0, 0, 0 }, { 0, 1, 1, 1 }, { 1, 0, 0, 0 }, { 1, 1, 1, 1 } };
	//vec_t XOR_y = { 0, 1, 1, 0 };
	
	//Mlp n(0.03, 0.01);
 
	/*拟合XOR参数：*/
	//n.add_layer(new FullyConnectedLayer(4, 10, new sigmoid_activation));
	//n.add_layer(new FullyConnectedLayer(10, 1, new sigmoid_activation));

	//n.train(XOR_x, XOR_y, 4);

	/*MNIST拟合：*/
	//n.add_layer(new FullyConnectedLayer(28 *28, 10, new sigmoid_activation));
#ifdef GENERATE_WEIGHT
	Mlp n(0.03, 0.01);
	n.add_layer(new FullyConnectedLayer(28 *28, 300, new sigmoid_activation));
	n.add_layer(new FullyConnectedLayer(300, 10, new sigmoid_activation));
	n.train(train_x, train_y, 60000);
	std::ofstream weight_file("weight.txt");
	n.fout_weight(weight_file); 
	n.test(test_x,test_y,10000);
#else
	//std::ifstream weight_file("weight.txt");
	//n.fin_weight(weight_file);
#endif

#ifdef GENERATE_FAULT
	//std::ofstream fault_file("fault.txt");
	Mlp n(0.03, 0.01);
	n.add_layer(new FullyConnectedLayer(28 *28, 300, new sigmoid_activation));
	n.add_layer(new FullyConnectedLayer(300, 10, new sigmoid_activation));
	std::ifstream weight_file("weight.txt");
	n.fin_weight(weight_file);
	for (int i = 0; i < 50;i++)
	{
		std::stringstream ss;
		ss << "fault7/fault" << i << ".txt";
		std::ofstream fault_file(ss.str());
		n.generateFault_varition(0.7);		
		n.fout_fault(fault_file);
		fault_file.close();
		std::ifstream fault_fileo(ss.str());
		n.fin_fault(fault_fileo);
		cout<<endl << i << ' '<< n.fault_test(test_x, test_y, 10000);
		fault_fileo.close();
	}
#endif

#ifdef GENERATE_FAULT_TYPE
	//std::ofstream fault_file("fault.txt");
	Mlp n(0.03, 0.01);
	n.add_layer(new FullyConnectedLayer(28 *28, 10, new sigmoid_activation));
	for (int i = 0; i < 100;i++)
	{
		std::stringstream ss;
		ss << "faultsa_new/fault" << i << ".txt";
		std::ofstream fault_file(ss.str());
		n.generateFault_sa();
		cout<<endl << i << ' ';
		float tmp = n.fault_test(test_x, test_y, 10000);
		cout<<tmp;
		n.fout_fault_type(fault_file);
		fault_file.close();
	}
#endif  

#if defined(REMAP) && !defined(GENERATE_FAULT)
	Mlp n(0.03, 0.01);
	n.add_layer(new FullyConnectedLayer(28 *28, 300, new sigmoid_activation));
	n.add_layer(new FullyConnectedLayer(300, 10, new sigmoid_activation));
	std::ifstream weight_file("weight.txt");
	n.fin_weight(weight_file);
	cout<<n.test(test_x, test_y, 10000)<<endl;
	/* for(int j=20;j<21;j+=2){              
		std::cout << "openfault:fault" << j << endl; */
	for (int i = 0; i < 20;i++)     
	{
		std::stringstream ss;
		//ss << "fault" << j <<"/fault" << i << ".txt";
		ss << "fault" << argv[1] <<"/fault" << i << ".txt";
		//ss << "fault5/fault" << i << ".txt";
		std::ifstream fault_file(ss.str());
		n.fin_fault(fault_file);
		/* std::stringstream sst;
		sst << "faultsa/fault" << i << ".txt";
		std::ifstream fault_file_t(sst.str());
		n.fin_fault_type(fault_file_t); */
		//n.generateFault_sa();
		cout<<n.fault_test(test_x, test_y, 10000)<<"\t";
		n.remap_best();
		cout<<n.fault_test(test_x, test_y, 10000);
		cout<<endl;
		fault_file.close();
		/* fault_file_t.close(); */
	}
	//}

#endif


#if defined(RETRAIN) && !defined(GENERATE_FAULT)

	
	for (int i =0;i < 5;i++)
	{
		Mlp n(0.03, 0.01);
		n.add_layer(new FullyConnectedLayer(28 *28, 300, new sigmoid_activation));
		n.add_layer(new FullyConnectedLayer(300, 10, new sigmoid_activation));
		/* n.add_layer(new FullyConnectedLayer(28 *28, 300, new sigmoid_activation));
		n.add_layer(new FullyConnectedLayer(300, 100, new sigmoid_activation));
		n.add_layer(new FullyConnectedLayer(100, 10, new sigmoid_activation)); */
		std::ifstream weight_file("weight.txt");
		n.fin_weight(weight_file);
		
		std::stringstream ss;
		ss << "fault" << argv[1] <<"/fault" << i << ".txt";
		std::cout<<std::endl << "openfault:" << ss.str() << endl;
		//ss << "fault16/fault" << i << ".txt";
		std::ifstream fault_file(ss.str());
		//std::ifstream fault_file("fault18/fault2.txt");
		n.fin_fault(fault_file);
	 	/* std::stringstream sst;
		sst << "faultsa_new/fault" << i << ".txt";
		std::ifstream fault_file_t(sst.str());
		n.fin_fault_type(fault_file_t); */
		cout<<n.test(test_x, test_y, 10000);
		cout<<" ";
		cout<<n.fault_test(test_x, test_y, 10000);
		cout<<" ";
		n.remap_best();
		cout<<n.fault_test(test_x, test_y, 10000);
		cout<<" ";
		/* n.fix_sa_all();
		cout << n.fault_test(test_x, test_y, 10000);
		cout << endl; */
		//cout <<endl<<"start retraining layer 1:"<<endl;
		double data[5]={0,0,0,0,0};
		double fix_factor = 0;
		for (int j = 0; j < 300; j++){
			//n.retrain(train_x, train_y, 60000,0.055,0,fix_factor);
			n.retrain(train_x, train_y, 60000,0.025,0,fix_factor);
			double accuracy = n.fault_test(test_x, test_y, 10000);
			cout<<j<<":"<< accuracy <<endl;
			if(accuracy>0.92){
				fix_factor = 0.5;
			}else if(accuracy > 0.932){
				fix_factor = 0.25;
			}else if(accuracy > 0.94){
				fix_factor = 0.125;
			}
			bool flag=false;
				if(j%5==0)
				{
					flag = true;
					cout<<"iter"<< j <<" ";
					data[4] = data[3];
					data[3] = data[2];
					data[2] = data[1];
					data[1] = data[0];
					data[0] = n.fault_test(test_x, test_y, 10000);
					float aver = (data[4] + data[3] + data[2] + data[1] + data[0]) / 5;
					for (int j = 0; j < 5; j++)
					{
						if (data[j] - aver > 0.00005 || data[j] - aver < -0.00005) {
							flag = false;
							break;
						}
					}
					cout<<data[0]<<endl;
				}
				if(flag && data[4] != 0)
				{
					cout<<"end at iter "<<j<<"  test rate :"<<data[0]<<endl;
					break;
				}
			
		}
		
		/* cout <<endl<<"start retraining layer 2:"<<endl;
		double data2[5]={0,0,0,0,0};
		for (int j = 0; j < 300; j++){
			n.retrain(train_x, train_y, 60000,0.055,2);
			//cout<<"j:"<<n.fault_test(test_x, test_y, 10000)<<endl;
				
			bool flag=false;
				if(j%5==0)
				{
					flag = true;
					cout<<"iter"<< j <<" ";
					data2[4] = data2[3];
					data2[3] = data2[2];
					data2[2] = data2[1];
					data2[1] = data2[0];
					data2[0] = n.fault_test(test_x, test_y, 10000);
					float aver = (data2[4] + data2[3] + data2[2] + data2[1] + data2[0]) / 5;
					for (int j = 0; j < 5; j++)
					{
						if (data2[j] - aver > 0.005 || data2[j] - aver < -0.005) {
							flag = false;
							break;
						}
					}
					cout<<data2[0]<<endl;
				}
				if(flag && data2[4] != 0)
				{
					cout<<endl<<"layer 2 end at iter "<<j<<"  "<<data2[0]<<endl;
					break;
				}
			
		} */
		
		//cout<<n.fault_test(test_x,test_y,10000);
		cout<<endl<<endl;
		fault_file.close();
		/* fault_file_t.close(); */
		weight_file.close();

	}
 
#endif
	//std::cout << "mean_fault_error:" << mean_error_rate / 50 << std::endl
	//weight_file.close();

#if /*defined(GENERATE_FAULT) ||*/ defined(RETRAIN)
//	fault_file.close();
#endif
	
	//ofile.close();
#if defined(_WIN32) || defined(_WIN64)
	getchar();
#endif
	return 0;
}
