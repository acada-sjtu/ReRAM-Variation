#define MAIN2
#ifdef MAIN2 
#include "network.h"

#include "fstream"
#include "iostream"
#include <string>
#include <sstream>

using namespace mlp;
using namespace std;

#define RETRAIN

int main(){
	vec2d_t train_x;
	vec_t train_y;
	vec2d_t test_x;
	vec_t test_y;
	//std::ofstream ofile("weight.txt");
	/*加载MNIST数据集*/
	LOAD_MNIST_TEST(test_x, test_y);
	LOAD_MNIST_TRAIN(train_x, train_y);
	
	Mlp n(0.03, 0.01);

	n.add_layer(new FullyConnectedLayer(28 *28, 10, new sigmoid_activation));
	
	for(int i=18;i<=18;i++){
		std::ifstream weight_file("weight.txt");
		n.fin_weight(weight_file);
		std::stringstream ss;
		ss << "generate_fault" << i << ".log";
		std::cout<<ss.str()<<std::endl;
		std::ifstream generate_logfile(ss.str());
		int j;
		j=1;
		/* for(j=0;j<500;j++)
		{
			float fault_;
			generate_logfile>>index>>fault_;
			if(fault_>0.43+(8-i)*0.05 && fault_<0.46+(8-i)*0.05)break;
			
		} 
		std::cout<<j<<' '<<index<<std::endl;*/
		if (i!=501){
			ss.clear();
			ss.str("");
			ss << "fault" << i << "/fault" << j << ".txt";
			std::ifstream fault_file(ss.str());
			std::cout<<ss.str()<<std::endl;
			n.fin_fault(fault_file);

			ss.clear();
			ss.str("");
			ss << "retrain" << i << "/fault" << j << ".logN50I100";
			std::cout<<"logfile:"<<ss.str()<<std::endl;
			std::ofstream fault_log(ss.str());

			ss.clear();
			ss.str("");
			ss << "retrain" << i << "/fault" << j << ".nwN50I100";
			std::cout << "new weight file:" << ss.str() << std::endl;
			std::ofstream weight_file(ss.str());

			ss.clear();
			ss.str("");
			ss << "retrain" << i << "/fault" << j << ".fwN50I100";
			std::cout << "fixed file:" << ss.str() << std::endl;
			std::ofstream fix_file(ss.str());

			std::streambuf *coutbuf = std::cout.rdbuf();
			std::cout.rdbuf(fault_log.rdbuf());
			n.test(test_x, test_y, 10000);
			n.fault_test(test_x, test_y, 10000);
			n.remap_best();
			n.fault_test(test_x, test_y, 10000);
			for (int i = 0; i < 250; i++){
				n.retrain(train_x, train_y, 60000,0.05);
				n.fault_test(test_x, test_y, 10000);
			}
			n.fout_weight(weight_file);
			n.fout_fixed(fix_file);

			weight_file.close();
			fix_file.close();

			fault_file.close();
			std::cout.rdbuf(coutbuf);
		}
		weight_file.close();
		generate_logfile.close();
		
	}

	return 0;
}
#endif
