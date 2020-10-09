#ifndef NETWORK_H_
#define NETWORK_H_

#pragma once

#include "util.h"
#include "mnist_parser.h"
#include "output_layer.h"
#include "mnist_parser.h"
#include "fullyconnected_layer.h"
#include "fstream"
#include "iostream"
#include <math.h>

namespace mlp{
#define MAX_ITER 300
#define FIX_NUMBER 25
#define FIX_NUMBER_1 750
#define FIX_NUMBER_2 10
/* #define FIX_NUMBER_1 750
#define FIX_NUMBER_2 95
#define FIX_NUMBER_3 3 */
#define M 50
#define END_CONDITION 1e-2
#define MIN_FAULT -2*3
#define MAX_FAULT 2*3
#define QUANT_BIT 7
//#define RETRAIN_ALL

	class Mlp
	{
	public:
		Mlp(float_t alpha, float_t lambda):
			alpha_(alpha), lambda_(lambda)
		{}

		void retrain(const vec2d_t& train_x, const vec_t& train_y, size_t train_size,double end_condition, int layer_num, double fix_factor){
			train_x_ = train_x;
			train_y_ = train_y;
			train_size_ = train_size;

			this->add_layer(new OutputLayer(layers.back()->out_depth_, layers.back()->a_));

			/* for (auto layer : layers){
				layer->alpha_ = alpha_, layer->lambda_ = lambda_;
				layer->find_fixed(FIX_NUMBER);
			}
			 */
			if (layer_num == 1) 
			{
				layers[0]->alpha_ = alpha_, layers[0]->lambda_ = lambda_;
				layers[0]->find_fixed(FIX_NUMBER_1,fix_factor);
			}
			else if (layer_num == 2)
			{
				layers[1]->alpha_ = alpha_, layers[1]->lambda_ = lambda_;
				layers[1]->find_fixed(FIX_NUMBER_2,fix_factor);
			}
			else 
			{
				//find_fixed_all(500);
				layers[0]->alpha_ = alpha_, layers[0]->lambda_ = lambda_;
				layers[0]->find_fixed(FIX_NUMBER_1,fix_factor);
				layers[1]->alpha_ = alpha_, layers[1]->lambda_ = lambda_;
				layers[1]->find_fixed(FIX_NUMBER_2,fix_factor); 
				/* for (auto layer : layers){
				layer->alpha_ = alpha_, layer->lambda_ = lambda_;
				layer->find_fixed(FIX_NUMBER);
				} */
			}

			auto stop = false;
			int iter = 0;
			vec_index train_array;
			std::srand(unsigned(time(NULL)));

#ifdef RETRAIN_ALL
			for (int i = 0; i < train_size; i++)train_array.push_back(i);
#endif
			while (iter < MAX_ITER && !stop){
				iter++;
#ifdef RETRAIN_ALL
				std::random_shuffle(train_array.begin(), train_array.end());
#else
				train_array.clear();
				for (int i = 0; i < 1000; i++)train_array.push_back(uniform_rand(0, train_size_ - 1));
#endif
				auto err = retrain_once(train_array);
				//std::cout << "err: " << err << std::endl;
				if (err < end_condition) stop = true;
			}
			this->delete_layer();
		}

		void find_fixed_all(int number){
			std::vector<int> index_r,index_c,index_l,isW;
			index_r.resize(number);
			index_c.resize(number);
			index_l.resize(number);
			isW.resize(number);
			std::vector<float> max_variation;
			max_variation.resize(number);
			for (int i = 0; i < number; i++){
				max_variation[i] = 0;
			}
			double swv;
			for(int l=0; l<2; l++){
				for (int i = 0; i < layers[l]->out_depth_; i++)
				{
					/************************  W_ *********************************/
					for (int j = 0; j < layers[l]->in_depth_; j++){
						//calculate swv
						if (layers[l]->Fault_.getFaultType(i, j)== 2)//2-variation
						{
							swv = fabs(layers[l]->Fault_.getFaultValue(i, j) * layers[l]->W_[i*layers[l]->in_depth_ + j]);
							if (swv > max_variation[number - 1]){
								int tmp = number - 1;
								for (; tmp>0 && swv > max_variation[tmp - 1]; tmp--);
								for (int k = number - 1; k > tmp; k--)
								{
									max_variation[k] = max_variation[k - 1];
									index_r[k] = index_r[k - 1];
									index_c[k] = index_c[k - 1];
									index_l[k] = index_l[k - 1];
									isW[k] = isW[k - 1];
								}
								max_variation[tmp] = swv;
								index_r[tmp] = i;
								index_c[tmp] = j;
								index_l[tmp] = l;
								isW[tmp] = 1;
							}
						}
					}
					/************************  b_ *********************************/
					//calculate swv
						if (layers[l]->Fault_.getFaultType(i, layers[l]->in_depth_)== 2)//2-variation
						{
							swv = fabs(layers[l]->Fault_.getFaultValue(i, layers[l]->in_depth_) * layers[l]->b_[i]);
							if (swv > max_variation[number - 1]) {
								int tmp = number - 1;
								for (; tmp > 0 && swv > max_variation[tmp - 1]; tmp--);
								for (int k = number - 1; k > tmp; k--)
								{
									max_variation[k] = max_variation[k - 1];
									index_r[k] = index_r[k - 1];
									index_c[k] = index_c[k - 1];
									index_l[k] = index_l[k - 1];
									isW[k] = isW[k - 1];
								}
								max_variation[tmp] = swv;
								index_r[tmp] = i;
								index_c[tmp] = 0;
								index_l[tmp] = l;
								isW[tmp] = 0;
							}
						}
				}
			}
			for (int i = 0; i < number; i++){
				if (isW[i] == 1)
				{
					layers[index_l[i]]->W_[index_r[i] * layers[index_l[i]]->in_depth_ + index_c[i]] = 0;
					layers[index_l[i]]->W_fix[index_r[i] * layers[index_l[i]]->in_depth_ + index_c[i]] = 1;
				}
				else
				{
					layers[index_l[i]]->b_[index_r[i]] = 0;
					layers[index_l[i]]->b_fix[index_r[i]] = 1;
				}
			}
		}
		
		void fix_sa_all(){
			for (auto layer : layers){
				layer->fix_sa();
			}
		}

		void generateFault_varition(float sigma){
			for (auto layer : layers){
				layer->generateFault_varition(sigma);
			}
		}
		
		void generateFault_sa(){
			for (auto layer : layers){
				layer->generateFault_sa();
			}
		}
		
		void remap_best(){
			for (auto layer : layers){
				layer->remap_best();
			}
			//layers[1]->remap_best();
		}

		void train(const vec2d_t& train_x, const vec_t& train_y, size_t train_size){
			train_x_ = train_x;
			train_y_ = train_y;
			train_size_ = train_size;
			/*
			auto add OutputLayer as the last layer.
			*/
			this->add_layer(new OutputLayer(layers.back()->out_depth_, layers.back()->a_));

			for (auto layer : layers){
				layer->alpha_ = alpha_, layer->lambda_ = lambda_;
			}

			/*
			start training...
			*/
			auto stop = false;
			int iter = 0;
			vec_index train_array;
			std::srand(unsigned(time(NULL)));

			for (int i = 0; i < train_size; i++)train_array.push_back(i);
			
			while (iter < MAX_ITER && !stop){
				iter++;
				std::random_shuffle(train_array.begin(), train_array.end());
				auto err = train_once(train_array);
				std::cout << "err: " <<  err << std::endl;
				if (err < END_CONDITION) stop = true;
			}
			this->delete_layer();
		}

		float test(const vec2d_t& test_x, const vec_t& test_y, size_t test_size){
			test_x_ = test_x, test_y_ = test_y, test_size_ = test_size;
			int iter = 0;
			int bang = 0;
			while (iter < test_size_){
				if (test_once(iter)) bang++;
				iter++;
			}
			//std::cout << (float)bang / test_size_ << std::endl;
			return (float)bang / test_size_;
		}

		float fault_test(const vec2d_t& test_x, const vec_t& test_y, size_t test_size){
			test_x_ = test_x, test_y_ = test_y, test_size_ = test_size;
			int iter = 0;
			int bang = 0;
			while (iter < test_size_){
				if (fault_test_once(iter)) bang++;
				iter++;
			}
			//std::cout << (float)bang / test_size_ << std::endl;
			return (float)bang / test_size_;
		}

		void add_layer(Layer* layer){
			if (!layers.empty())
				this->layers.back()->next = layer;
			this->layers.push_back(layer);
			layer->next = NULL;
		}

		void delete_layer(){
			if (!layers.empty()){
				auto layer = layers[0];
				if (layer->next == NULL)return;
				else while (layer->next->next != NULL)layer = layer->next;
				layer->next = NULL;
				this->layers.pop_back();
			}
		}

		void fout_weight(std::ofstream &ofile)
		{
			for (auto layer : layers){
				ofile << layer->in_depth_ << " " << layer->out_depth_ << std::endl;
				for (int i = 0; i < layer->out_depth_; i++){
					for(int j = 0; j < layer->in_depth_; j++)
					{
						ofile << layer->W_[i*layer->in_depth_ + j] << " ";
					}
					ofile << layer->b_[i] << std::endl;
				}
			}
		}

		void fin_weight(std::ifstream &infile)
		{
			for (auto layer : layers)
			{
				int in_depth, out_depth;
				infile >> in_depth >> out_depth;
				for (int i = 0; i < out_depth; i++){
					for (int j = 0; j < in_depth;j++)
					{
						infile >> layer->W_[i*in_depth + j];
					//	layer->W_[i*in_depth + j]=quantW(layer->W_[i*in_depth + j],QUANT_BIT);
					}
					infile >> layer->b_[i];
					//layer->b_[i]=quantW(layer->b_[i],QUANT_BIT);
				}
			}
			
		}

		void fout_fixed(std::ofstream &ofile)
		{
			for (auto layer : layers){
				ofile << layer->in_depth_ << " " << layer->out_depth_ << std::endl;
				for (int i = 0; i < layer->out_depth_; i++){
					for (int j = 0; j < layer->in_depth_ + 1; j++)
					{
						ofile << (int)(layer->W_fix[i*layer->in_depth_+j]) << " ";
					}
					ofile << (int)(layer->b_fix[i]) << std::endl;
				}
			}
		}

		void fout_fault(std::ofstream &ofile)
		{
			for (auto layer : layers){
				ofile << layer->in_depth_ << " " << layer->out_depth_ << std::endl;
				for (int i = 0; i < layer->out_depth_; i++){
					for (int j = 0; j < layer->in_depth_+1; j++)
					{
						ofile << layer->Fault_.getFaultValue(i,j) << " ";
					}
				}
			}
		}

		void fin_fault(std::ifstream &infile)
		{
			for (auto layer : layers)
			{
				int in_depth, out_depth;
				float tmp;
				infile >> in_depth >> out_depth;
				for (int i = 0; i < out_depth; i++){
					for (int j = 0; j < in_depth+1; j++)
					{
						infile >> tmp;
						layer->Fault_real_.setFaultValue(i, j, -tmp);
						//tmp = quantF(-tmp,QUANT_BIT);
						layer->Fault_.setFaultValue(i, j, -tmp);
						layer->Fault_.setFaultType(i, j, 2);
					}
				}
			}
		}
		
		void fout_fault_type(std::ofstream &ofile)
		{
			for (auto layer : layers){
				ofile << layer->in_depth_ << " " << layer->out_depth_ << std::endl;
				for (int i = 0; i < layer->out_depth_; i++){
					for (int j = 0; j < layer->in_depth_+1; j++)
					{
						ofile << layer->Fault_.getFaultType(i,j) << " ";
					}
				}
			}
		}
		
		void fin_fault_type(std::ifstream &infile)
		{
			for (auto layer : layers)
			{
				int in_depth, out_depth;
				int tmp;
				infile >> in_depth >> out_depth;
				for (int i = 0; i < out_depth; i++){
					for (int j = 0; j < in_depth+1; j++)
					{
						infile >> tmp;
						layer->Fault_.setFaultType(i, j, tmp);
					}
				}
			}
		}

	private:
		size_t max_iter(const vec_t& v){
			size_t i = 0;
			float_t max = v[0];
			for (size_t j = 1; j < v.size(); j++){
				if (v[j] > max){
					max = v[j];
					i = j;
				}
			}
			return i;
		}

		vec_t quantization(const vec_t& in,int n){
			vec_t v;
			int size = in.size();
			for(int i=0;i<size;i++){
				float q=1/pow(2,n);
				float min=0;
				float max=q;
				for(int j=0;j<pow(2,n);j++){
					if(in[i]>=min && in[i]<max)
					{
						v.push_back(min);
						//v.push_back(0.5*(min+max));
						break;
					}
					else if(in[i]==1)
					{
						v.push_back(1);
						break;
					}
					min+=q;
					max+=q;
				}
			
			}
			return v;
		}
		
		
		
		float quantF(float x,int n){
			/* float y;
			float q=(100+1)/pow(2,n);
			float min=-1;
			float max=-1+q;
			for(int j=0;j<pow(2,n);j++){
				if(x>=min && x<max)
				{
					//y=min;
					y= 0.5*(min+max);
					break;
				}
				min+=q;
				max+=q;
			}
			if(x >= MAX_FAULT)
				{
					y= 100;
					
				}
		   // std::cout <<x<<"->"<<y << "  ";
			return y; */
			
			float y;
			float x_log = log(x+double(1));
			float q=(MAX_FAULT-MIN_FAULT)/pow(2,n);
			float min=MIN_FAULT;
			float max=MIN_FAULT+q;
			for(int j=0;j<pow(2,n);j++){
				if(x_log>=min && x_log<max)
				{
					// y=min;
					y= exp(0.5*(min+max))-1.0;
					break;
				}
				min+=q;
				max+=q;
			}
			if(x_log >= MAX_FAULT)
				{
					y= exp(MAX_FAULT)-1.0;
					
				}
		   // std::cout <<x<<"->"<<y << "  ";
			return y;
		}
		
		bool test_once(int index){
			//auto test_x_index = uniform_rand(0, test_size_ - 1);
			int test_x_index = index;
			layers[0]->input_ = test_x_[test_x_index];
			for (auto layer : layers){
				layer->forward();
				if (layer->next != nullptr){
					layer->next->input_ = layer->output_;
				}
			}
			//std::cout << "exp:" << test_y_[test_x_index];
			//std::cout << "result:";
			//disp_vec_t(layers.back()->output_);
			//return true;
			return (int)test_y_[test_x_index] == (int)max_iter(layers.back()->output_);
		}

		bool fault_test_once(int index){
			//auto test_x_index = uniform_rand(0, test_size_ - 1);
			int test_x_index = index;
			layers[0]->input_ = test_x_[test_x_index];
			//layers[0]->input_ = quantization(test_x_[test_x_index],QUANT_BIT);
			for (auto layer : layers){
				layer->fault_forward();
				if (layer->next != nullptr){
					layer->next->input_ = layer->output_;
				}
			}
			/* std::cout << "exp:" << test_y_[test_x_index];
			std::cout << "result:"<<(int)max_iter(quantization(layers.back()->output_,QUANT_BIT))<<std::endl;
			disp_vec_t(quantization(layers.back()->output_,QUANT_BIT)); */
			//return true;
			return (int)test_y_[test_x_index] == (int)max_iter(layers.back()->output_);
			//return (int)test_y_[test_x_index] == (int)max_iter(quantization(layers.back()->output_,QUANT_BIT));
		}

		float_t retrain_once(vec_index vec){
			float_t err = 0;
			int size = vec.size();
			int iter = 0;
			while (iter < size){
				//auto train_x_index = uniform_rand(0, train_size_ - 1);
				assert(vec[iter] < train_size_);
				int train_x_index = vec[iter];
				layers[0]->input_ = train_x_[train_x_index];
				layers.back()->exp_y = (int)train_y_[train_x_index];

				/*期待结果*/
				//std::cout << "layer exp y: " << layers.back()->exp_y << std::endl;
				/*
				Start forward feeding.
				*/
				for (auto layer : layers){
					layer->forward();
					//layer->fault_forward();
					if (layer->next != nullptr){
						layer->next->input_ = layer->output_;
					}
				}

				/*MNIST 每一轮拟合后的结果*/
				//std::cout << (int)max_iter(layers.back()->input_) << std::endl;

				/*输出XOR每一轮拟合后的结果*/
				//disp_vec_t(layers.back()->input_);

				err += layers.back()->err;
				/*
				back propgation
				*/

				for (auto i = layers.rbegin(); i != layers.rend(); i++){
					(*i)->fix_backprop();
				}
				iter++;
			}
			return err / size;
		}

		float_t train_once(vec_index vec){
			float_t err = 0;
			int size = vec.size();
			int iter = 0;
			while (iter < size){
				//auto train_x_index = uniform_rand(0, train_size_ - 1);
				assert(vec[iter] < train_size_);
				int train_x_index = vec[iter];
				layers[0]->input_ = train_x_[train_x_index];
				layers.back()->exp_y = (int)train_y_[train_x_index];
				
				/*期待结果*/
				//std::cout << "layer exp y: " << layers.back()->exp_y << std::endl;
				/*
				Start forward feeding.
				*/
				for (auto layer : layers){
					layer->forward();
					if (layer->next != nullptr){
						layer->next->input_ = layer->output_;
					}
				}

				/*MNIST 每一轮拟合后的结果*/
				//std::cout << (int)max_iter(layers.back()->input_) << std::endl;
				
				/*输出XOR每一轮拟合后的结果*/
				//disp_vec_t(layers.back()->input_);

				err += layers.back()->err;
				/*
				back propgation
				*/

				for (auto i = layers.rbegin(); i != layers.rend(); i++){
					(*i)->back_prop();
				}
				iter++;
			}
			return err / size;
		}

		std::vector < Layer* > layers;

		size_t train_size_;
		vec2d_t train_x_;
		vec_t train_y_;

		size_t test_size_;
		vec2d_t test_x_;
		vec_t test_y_;

		float_t alpha_;
		float_t lambda_;
	};
#undef MAX_ITER
#undef M
} //namespace mlp

#endif
