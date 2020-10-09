#ifndef LAYER_H_
#define LAYER_H_
#pragma once
#include "activation.h"
#include "fault.h"

namespace mlp{
	class Layer
	{
	public:
		Layer(size_t in_depth,
			size_t out_depth, activation* a) :
			in_depth_(in_depth), out_depth_(out_depth), a_(a)
		{}

		virtual void init_weight() = 0;
		virtual void forward() = 0;
		virtual void back_prop() = 0;
		virtual void fix_backprop() = 0;
		virtual void fault_forward() = 0;
		virtual void generateFault_varition(float sigma) = 0;
		virtual void generateFault_sa() = 0;
		virtual void find_fixed(int number, double fix_factor) = 0;
		virtual void fix_sa() = 0;
		/* virtual void remap() = 0;
		virtual void remap_last() = 0; */
		virtual void remap_best() = 0;

		
		size_t in_depth_;
		size_t out_depth_;

		vec_t W_;
		vec_t deltaW_; //last iter weight change for momentum;

		vec_t b_;

		Fault Fault_;
		Fault Fault_real_;

		vec_char W_fix;
		vec_char b_fix;
		size_t fixed_number;

		vec_t F_;

		activation* a_;

		vec_t input_;
		vec_t output_;

		Layer* next;

		float_t alpha_; // learning rate
		float_t lambda_; // momentum
		vec_t g_; // err terms

		/*output*/
		float_t err;
		int exp_y;
		vec_t exp_y_vec;
	};
} //namspace mlp

#endif