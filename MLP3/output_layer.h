#ifndef OUTPUT_LAYER_H_
#define OUTPUT_LAYER_H_

#pragma once

#include "layer.h"
#include "util.h"

namespace mlp{
	class OutputLayer :public Layer
	{
	public:
		OutputLayer(size_t in_depth, activation* a) :
			Layer(in_depth, 0, a)
		{}

		void forward(){
			this->err = 0;
			exp_y_vec.clear();
			/*XOR 使用下面一行代码*/
			//exp_y_vec.push_back(this -> exp_y);

			/*MNIST 使用下面两行代码 */
			exp_y_vec.resize(in_depth_);
			for (size_t i = 0; i < exp_y_vec.size(); i++)exp_y_vec[i] = 0;
			exp_y_vec[this->exp_y] = 1;

			for (size_t i = 0; i < in_depth_; i++){
				err += 0.5 * (exp_y_vec[i] - input_[i]) *
					(exp_y_vec[i] - input_[i]);
			}
			output_ = input_;
		}

		void back_prop(){
			/* compute err terms of output layers */
			g_.clear();

			for (size_t i = 0; i < in_depth_; i++){
				g_.push_back((exp_y_vec[i] - input_[i]) * a_->df(input_[i]));
			}
		}

		void fix_backprop(){
			g_.clear();

			for (size_t i = 0; i < in_depth_; i++){
				g_.push_back((exp_y_vec[i] - input_[i]) * a_->df(input_[i]));
			}
		}

		void init_weight(){}

		void fault_forward(){
			this->err = 0;
			exp_y_vec.clear();
			/*XOR 使用下面一行代码*/
			//exp_y_vec.push_back(this -> exp_y);

			/*MNIST 使用下面两行代码 */
			exp_y_vec.resize(in_depth_);
			for (size_t i = 0; i < exp_y_vec.size(); i++)exp_y_vec[i] = 0;
			exp_y_vec[this->exp_y] = 1;

			for (size_t i = 0; i < in_depth_; i++){
				err += 0.5 * (exp_y_vec[i] - input_[i]) *
					(exp_y_vec[i] - input_[i]);
			}
			output_ = input_;
		}

		void generateFault_varition(float s){}
		
		void generateFault_sa(){}

		void find_fixed(int number, double fix_factor){}

		void fix_sa(){}

		void remap(){}
		
		void remap_last(){}
		
		void remap_best(){}
		
	private:
	};
} // namespace mlp

#endif