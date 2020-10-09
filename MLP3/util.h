#ifndef UTIL_H_
#define UTIL_H_

#pragma once

#include <vector>
#include <iostream>
#include <cstdint>
#include <cassert>
#include <time.h>
#include <stdlib.h>

#include "boost/random.hpp"

namespace mlp {
	#define MAX_WEIGHT 7.13
	#define MIN_WEIGHT 0
	typedef std::vector<float_t> vec_t;
	typedef std::vector<std::vector<float_t>> vec2d_t;
	typedef std::vector<char> vec_char;
	typedef std::vector<std::vector<char>> vec2d_char;
	typedef std::vector<int> vec_index;

	struct fault
	{
		int type;  //0-sa0;  1-sa1;   2-variation
		double value; //value of variation
	};

	typedef std::vector<fault> vec_fault;
	typedef std::vector<std::vector<fault>> vec2d_fault;

	inline int uniform_rand(int min, int max) {
		static boost::mt19937 gen(0);
		boost::uniform_smallint<> dst(min, max);
		return dst(gen);
	}

	template<typename T>
	inline T uniform_rand(T min, T max) {
		static boost::mt19937 gen(0);
		boost::uniform_real<T> dst(min, max);
		return dst(gen);
	}

	template<typename Iter>
	void uniform_rand(Iter begin, Iter end, float_t min, float_t max) {
		for (Iter it = begin; it != end; ++it)
			*it = uniform_rand(min, max);
	}

	void disp_vec_t(const vec_t& v){
		for (auto i : v)
			std::cout << i << "\t";
		std::cout << "\n";
	}

	void disp_vec2d_t(const vec2d_t& v){
		for (auto i : v){
			for (auto i_ : i)
				std::cout << i_ << "\t";
			std::cout << "\n";
		}
	}

	/* float quantW(float x,int n){
			float y;
			int sign;
			float tmp;
			if(x<0)
			{
				sign=1;
				tmp=-x;
			} //negative
			else 
			{
				sign=0;
				tmp = x;
			}
			
			float q=(MAX_WEIGHT-MIN_WEIGHT)/pow(2,n);
			float min=MIN_WEIGHT;
			float max=MIN_WEIGHT+q;
			for(int j=0;j<pow(2,n);j++){
				if(tmp>=min && tmp<max)
				{
					//y=min;
					y= 0.5*(min+max);
					break;
				}
				min+=q;
				max+=q;
			}
		
		if(sign==1){ y=-y; }
             //   std::cout <<x<<"->"<<tmp <<"->"<<y << "  ";
			return y;
		} */
	
	float_t fault_dot(const vec_t& x, const vec_t& w,const vec_t& f, const vec_char& ft){
		assert(x.size() == w.size());
		assert(x.size() == f.size()); 
		assert(f.size() == ft.size());
		float_t sum = 0;
		for (size_t i = 0; i < x.size(); i++){
			if(ft[i]==2)//2-variation
				{
					if(w[i] * f[i] > MAX_WEIGHT)
					{
						sum += x[i] * MAX_WEIGHT;
					}
					else {sum += x[i] * w[i] * f[i];}
				}
			else if (ft[i]==0)  //0-sa0
				{sum += x[i] * MAX_WEIGHT;}
			else if (ft[i] == 3)
			{
				sum += x[i] * (-MAX_WEIGHT);
			}
			else if (ft[i]==1)//1-sa1
				{sum += x[i] * MIN_WEIGHT;}
		}
		return sum;
	}

	float_t fault_b(const float t, const float fault, const int fault_type)
	{
		if (fault_type == 2)return t*(1 + fault);
		if (fault_type == 1)return MIN_WEIGHT;
		if (fault_type == 0)return MAX_WEIGHT;
		if (fault_type == 3)return -MAX_WEIGHT;
		return 0;

	}
	
	float_t dot(const vec_t& x, const vec_t& w){
		assert(x.size() == w.size());
		float_t sum = 0;
		for (size_t i = 0; i < x.size(); i++){
			sum += x[i] * w[i];
		}
		return sum;
	}
	
	/********************** added  ************************/	
	float_t abs_dot(const vec_t& w, const vec_t& f,const vec_char& ft){
		assert(w.size() == f.size());
		assert(f.size() == ft.size());
		float_t sum = 0;
		for (size_t i = 0; i < w.size(); i++){
			if (ft[i]==2)//2-variation
				{sum += fabs(w[i] * f[i]);}
			else if(ft[i]==0)//0-sa0
				{sum += fabs(MAX_WEIGHT - w[i]);}
			else if(ft[i]==1)//1-sa1
				{sum += fabs(MIN_WEIGHT - w[i]);}
			else if (ft[i] == 3)
			{
				sum += fabs(-MAX_WEIGHT - w[i]);
			}
		}
		return sum;
	}
	
	vec_t f_muti_vec(float_t x, const vec_t& v){
		vec_t r;
		for_each(v.begin(), v.end(), [&](float_t i){
			r.push_back(x * i);
		});
		return r;
	}

	vec_t get_W(size_t index, size_t in_size_, const vec_t& W_){
		vec_t v;
		for (int i = 0; i < in_size_; i++){
			v.push_back(W_[index * in_size_ + i]);
		}
		return v;
	}
} // namespace mlp

#endif //UTIL_H_
