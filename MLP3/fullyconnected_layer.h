#ifndef FULLYCONNECTEDLAYER_H_
#define FULLYCONNECTEDLAYER_H_

#pragma once
#include "layer.h"
#include "util.h"
#include <math.h>
#include <vector>

#define MAX_WEIGHT 7.13
#define MIN_WEIGHT 0

namespace mlp{

	class FullyConnectedLayer :public Layer
	{
	public:
		FullyConnectedLayer(size_t in_depth, size_t out_depth, activation* a) :
			Layer(in_depth, out_depth, a)
		{
			output_.resize(out_depth_);
			W_.resize(in_depth_ * out_depth_);
			W_fix.resize(in_depth_ * out_depth_);
			for (int i = 0; i < in_depth_ * out_depth_; i++)
			{
				W_fix[i] = 0;
			}
			deltaW_.resize(in_depth_ * out_depth_);
			b_.resize(out_depth_);
			b_fix.resize(out_depth_);
			for (int i = 0; i < out_depth_; i++)
			{
				b_fix[i] = 0;
			}
			g_.resize(in_depth_);
			Fault_.resizeFault(out_depth_, in_depth_+1);
			Fault_real_.resizeFault(out_depth_, in_depth_+1);
			this->init_weight();
		}

		void find_fixed(int number, double fix_factor){
			std::vector<int> index_r,index_c,isW;
			index_r.resize(number);
			index_c.resize(number);
			isW.resize(number);
			std::vector<float> max_variation;
			max_variation.resize(number);
			for (int i = 0; i < number; i++){
				max_variation[i] = 0;
			}
			double swv;
			for (int i = 0; i < out_depth_; i++)
			{
				/************************  W_ *********************************/
				for (int j = 0; j < in_depth_; j++){
					//calculate swv
					if (Fault_.getFaultType(i, j)== 2)//2-variation
					{
						swv = fabs(Fault_.getFaultValue(i, j) * W_[i*in_depth_ + j]);
						if (swv > max_variation[number - 1]){
							int tmp = number - 1;
							for (; tmp>0 && swv > max_variation[tmp - 1]; tmp--);
							for (int k = number - 1; k > tmp; k--)
							{
								max_variation[k] = max_variation[k - 1];
								index_r[k] = index_r[k - 1];
								index_c[k] = index_c[k - 1];
								isW[k] = isW[k - 1];
							}
							max_variation[tmp] = swv;
							index_r[tmp] = i;
							index_c[tmp] = j;
							isW[tmp] = 1;
						}
					}
				}
				/************************  b_ *********************************/
				//calculate swv
					if (Fault_.getFaultType(i, in_depth_)== 2)//2-variation
					{
						swv = fabs(Fault_.getFaultValue(i, in_depth_) * b_[i]);
						if (swv > max_variation[number - 1]) {
							int tmp = number - 1;
							for (; tmp > 0 && swv > max_variation[tmp - 1]; tmp--);
							for (int k = number - 1; k > tmp; k--)
							{
								max_variation[k] = max_variation[k - 1];
								index_r[k] = index_r[k - 1];
								index_c[k] = index_c[k - 1];
								isW[k] = isW[k - 1];
							}
							max_variation[tmp] = swv;
							index_r[tmp] = i;
							index_c[tmp] = 0;
							isW[tmp] = 0;
						}
					}
			}
			for (int i = 0; i < number; i++){
				if (isW[i] == 1)
				{
					//W_[index_r[i] * in_depth_ + index_c[i]] /= 2;
					W_[index_r[i] * in_depth_ + index_c[i]] *= fix_factor;
					W_fix[index_r[i] * in_depth_ + index_c[i]] = 1;
				}
				else
				{
					//b_[index_r[i]] /= 2;
					b_[index_r[i]] *= fix_factor;
					b_fix[index_r[i]] = 1;
				}
			}
		}

		void fix_sa(){
			for (int i = 0; i < out_depth_; i++)
			{
				for (int j = 0; j < in_depth_; j++)
				{
					switch (Fault_.getFaultType(i, j)){
					case 0:
						W_[i*in_depth_ + j] = MAX_WEIGHT;
						W_fix[i*in_depth_ + j] = 1;
						break;
					case 1:
						W_[i*in_depth_ + j] = MIN_WEIGHT;
						W_fix[i*in_depth_ + j] = 1;
						break;
					case 3:
						W_[i*in_depth_ + j] = -MAX_WEIGHT;
						W_fix[i*in_depth_ + j] = 1;
						break;
					}
				}
				switch (Fault_.getFaultType(i, in_depth_)){
				case 0:
					b_[i] = MAX_WEIGHT;
					b_fix[i] = 1;
				case 1:
					b_[i] = MIN_WEIGHT;
					b_fix[i] = 1;
				case 3:
					b_[i] = -MAX_WEIGHT;
					b_fix[i] = 1;
				}

			}
		}


		void generateFault_varition(float sigma){
			Fault_.generateLogVariation(sigma);
			fixed_number = 0;
		}

		void generateFault_sa(){
			Fault_.generateSA0(0.0175,7);
			Fault_.generateSA1(0.0904,0,98);
			fixed_number = 0;
		}
		
		/* void remap(){
			double swv_min;
			double swv;
			int target=0;
			int *map = new int[in_depth_+1](); 
			int *v_queue = new int[in_depth_+1]();  //0-not map   1- mapped
			
			//consider b_ first
			for(size_t scan = 0; scan < in_depth_+1; scan++){
					swv_min = 10000;
					if(v_queue[scan] == 0){
								swv = abs_dot(b_,get_Fault_in(scan));
						if (swv < swv_min){
							swv_min = swv;
							target = scan;
						}
					}
				}
				v_queue[target] = 1; // mapped
				map[in_depth_] = target;
				
			for(size_t in = 0; in < in_depth_; in++){
				for(size_t scan = 0; scan < in_depth_+1; scan++){
					swv_min = 10000;
					if(v_queue[scan] == 0){
						swv = abs_dot(get_W_step(in),get_Fault_in(scan));
								
						if (swv < swv_min){
							swv_min = swv;
							target = scan;
						}
					}
				}
				v_queue[target] = 1; // mapped
				map[in] = target;
			}
			
			Fault temp;
			temp = Fault_;
			for(size_t in = 0; in < in_depth_+1; in++){
				for(size_t out = 0; out < out_depth_; out++){
					Fault_.setFaultValue(out,in,temp.getFaultValue(out,map[in]));
				}
			}
			delete[] map;
			delete[] v_queue;
		}
		
		void remap_last(){
			double swv_min;
			double swv;
			int target=0;
			int *map = new int[in_depth_+1](); 
			int *v_queue = new int[in_depth_+1]();  //0-not map   1- mapped
						
			//consider b_ last
			for(size_t in = 0; in < in_depth_+1; in++){
				for(size_t scan = 0; scan < in_depth_+1; scan++){
					swv_min = 10000;
					if(v_queue[scan] == 0){
								if (in!=in_depth_){
					          	swv = abs_dot(get_W_step(in),get_Fault_in(scan));}
								else {swv = abs_dot(b_,get_Fault_in(scan));}
						if (swv < swv_min){
							swv_min = swv;
							target = scan;
						}
					}
				}
				v_queue[target] = 1; // mapped
				map[in] = target;
			}
	
			Fault temp;
			temp = Fault_;
			for(size_t in = 0; in < in_depth_+1; in++){
				for(size_t out = 0; out < out_depth_; out++){
					Fault_.setFaultValue(out,in,temp.getFaultValue(out,map[in]));
				}
			}
			delete[] map;
			delete[] v_queue;
		} */
		
		void init (int size)
		{
			// 根据实际情况，添加代码以初始化
			n = size;
			for (int i = 0; i < n; i ++)
				for (int j = 0; j < n; j ++)
				  //  scanf ("%d", &weight [i] [j]);
				  //weight[i][j]=w[i][j];
				  if (i!=in_depth_){
					          	weight[i][j] = abs_dot(get_W_step(i),get_Fault_in(j),get_Fault_in_type(j));}
				 else {weight[i][j] = abs_dot(b_,get_Fault_in(j),get_Fault_in_type(j));}
	
		}

		bool path (int u)
		{
			sx [u] = true;
			for (int v = 0; v < n; v ++)
				if (!sy [v] && lx[u] + ly [v] == weight [u] [v])
					{
					sy [v] = true;
					if (match [v] == -1 || path (match [v]))
						{
						match [v] = u;
						return true;
						}
					}
			return false;
		}
		
		double bestmatch (bool maxsum)
		{
			int i, j;
			if (!maxsum)
				{
				for (i = 0; i < n; i ++)
					for (j = 0; j < n; j ++)
						weight [i] [j] = -weight [i] [j];
				}
			// 初始化标号
			for (i = 0; i < n; i ++)
				{
				lx [i] = -0x1FFFFFFF;
				ly [i] = 0;
				for (j = 0; j < n; j ++)
					if (lx [i] < weight [i] [j])
						lx [i] = weight [i] [j];
				}
			memset (match, -1, sizeof (match));
			for (int u = 0; u < n; u ++)
				while (1)
					{
					memset (sx, 0, sizeof (sx));
					memset (sy, 0, sizeof (sy));
					if (path (u))
						break;
					// 修改标号
					double dx = 0x7FFFFFFF;
					for (i = 0; i < n; i ++)
						if (sx [i])
							for (j = 0; j < n; j ++)
								if(!sy [j])
								{
									if(lx[i] + ly [j] - weight [i] [j] < dx)
										dx = lx[i] + ly [j] - weight [i] [j];
								}
									//dx = min (lx[i] + ly [j] - weight [i] [j], dx);
					for (i = 0; i < n; i ++)
						{
						if (sx [i])
							lx [i] -= dx;
						if (sy [i])
							ly [i] += dx;
						}
					}
			double sum = 0;
			for (i = 0; i < n; i ++)
			{
				sum += weight [match [i]] [i];
			}
			if (!maxsum)
				{
				sum = -sum;
				for (i = 0; i < n; i ++)
					for (j = 0; j < n; j ++)
						weight [i] [j] = -weight [i] [j];         // 如果需要保持 weight [ ] [ ] 原来的值，这里需要将其还原
				}
			return sum;
		}

		
		
		void remap_best(){
			init(in_depth_+1);
			double cost = bestmatch (false);
			Fault temp;
			Fault temp_real;
			temp = Fault_;
			temp_real = Fault_real_;
			for(size_t in = 0; in < in_depth_+1; in++){
				for(size_t out = 0; out < out_depth_; out++){
					Fault_.setFaultValue(out,match[in],temp.getFaultValue(out,in));
					Fault_real_.setFaultValue(out,match[in],temp_real.getFaultValue(out,in));
					Fault_.setFaultType(out,match[in],temp.getFaultType(out,in));
				}
			}
		
		}
		
		void fault_forward(){
			for (size_t out = 0; out < out_depth_; out++){
				output_[out] = a_->f(fault_dot(input_, get_W(out), get_Fault_real(out), get_Fault_Type(out)) + fault_b(b_[out], Fault_real_.getFaultValue(out, in_depth_), Fault_.getFaultType(out, in_depth_)));
			}
		}

		void forward(){
			for (size_t out = 0; out < out_depth_; out++){
				output_[out] = a_->f(dot(input_, get_W(out)) + b_[out]);
			}
		}

		void fix_backprop(){
			/*
			Compute the err terms;
			*/
			for (size_t in = 0; in < in_depth_; in++){
				g_[in] = a_->df(input_[in]) * dot(this->next->g_, get_W_step(in));
			}
			/*
			Update weights.
			*/
			for (size_t out = 0; out < out_depth_; out++){
				for (size_t in = 0; in < in_depth_; in++){
					if (W_fix[out * in_depth_ + in]==1)continue;
					auto delta = alpha_/*learning rate*/
						* input_[in] * this->next->g_[out]/*err terms*/
						/*lambda_ momentum*/
						+ lambda_ * deltaW_[out * in_depth_ + in];
					W_[out * in_depth_ + in] += delta;
					/*update momentum*/
					deltaW_[out * in_depth_ + in] = delta;
				}
				if(b_fix[out]==0)b_[out] += this->next->g_[out];
			}
		}

		void back_prop(){
			/*
			Compute the err terms;
			*/
			for (size_t in = 0; in < in_depth_; in++){
				g_[in] = a_->df(input_[in]) * dot(this->next->g_, get_W_step(in));
			}
			/*
			Update weights.
			*/
			for (size_t out = 0; out < out_depth_; out++){
				for (size_t in = 0; in < in_depth_; in++){
					auto delta = alpha_/*learning rate*/
						* input_[in] * this->next->g_[out]/*err terms*/
						/*lambda_ momentum*/
						+ lambda_ * deltaW_[out * in_depth_ + in];
					W_[out * in_depth_ + in] += delta;
					/*update momentum*/
					deltaW_[out * in_depth_ + in] = delta;
				}
				b_[out] += this->next->g_[out];
			}
		}

		/*
		for the activation sigmod,
		weight init ： [-4 * (6 / sqrt(fan_in + fan_out)), +4 *(6 / sqrt(fan_in + fan_out))]:
		see also:http://deeplearning.net/tutorial/references.html#xavier10
		*/
		void init_weight(){
			/*
			uniform_rand(W_.begin(), W_.end(),
			-4 * 6 / std::sqrtf((float)(fan_in() + fan_out())),
			4 * 6 / std::sqrtf((float)(fan_in() + fan_out())));
			uniform_rand(b_.begin(), b_.end(),
			-4 * 6 / std::sqrtf((float)(fan_in() + fan_out())),
			4 * 6 / std::sqrtf((float)(fan_in() + fan_out())));
			*/
			uniform_rand(W_.begin(), W_.end(), -1, 1);
			uniform_rand(b_.begin(), b_.end(), -1, 1);

		}
	private:
		double weight [1024] [1024];        // X 到 Y 的映射（权重）
		int n=in_depth_+1;
		double lx [1024], ly [1024];        // 标号
		bool sx [1024], sy [1024];    // 是否被搜索过
		int match [1024];        // Y(i) 与 X(match [i]) 匹配
		
		vec_t get_W(size_t index){
			vec_t v;
			for (int i = 0; i < in_depth_; i++){
				v.push_back(W_[index * in_depth_ + i]);
			}
			return v;
		}

		vec_t get_W_step(size_t in){
			vec_t r;
			for (size_t i = in; i < out_depth_ * (in_depth_); i += in_depth_){
				r.push_back(W_[i]);
			}
			return r;
		}
		
		vec_t get_Fault(size_t index){
			vec_t v;
			for (int i = 0; i < in_depth_; i++){
				v.push_back((Fault_.getFaultValue(index,i) + 1.0));
			}
			return v;
		}
		
		vec_t get_Fault_real(size_t index){
			vec_t v;
			for (int i = 0; i < in_depth_; i++){
				v.push_back((Fault_real_.getFaultValue(index,i) + 1.0));
			}
			return v;
		}
		
		vec_char get_Fault_Type(size_t index){
			vec_char v;
			for (int i = 0; i < in_depth_; i++){
				v.push_back(Fault_.getFaultType(index,i));
			}
			return v;
		}
		
		vec_t get_Fault_in(size_t index){
			vec_t v;
			for (int i = 0; i < out_depth_; i++){
				v.push_back(Fault_.getFaultValue(i,index));
			}
			return v;
		}
		
		vec_char get_Fault_in_type(size_t index){
			vec_char v;
			for (int i = 0; i < out_depth_; i++){
				v.push_back(Fault_.getFaultType(i,index));
			}
			return v;
		}
		
		
	};
} // namespace mlp

#endif 
