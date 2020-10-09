#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#pragma once
#include<algorithm>
namespace mlp {

	class activation {
	public:
		virtual float_t f(float_t x) const = 0;
		virtual float_t df(float_t f_x) const = 0;
	};

	class sigmoid_activation : public activation {
	public:
		float_t f(float_t x) const { return 1.0 / (1.0 + std::exp(-x)); }
		float_t df(float_t f_x) const { return f_x * (1.0 - f_x); }
	};
	
	class ReLU_activation : public activation {
	public:
		float_t f(float_t x) const { return std::max(float_t(0.0) , x); }
		float_t df(float_t f_x) const { 
			if (f_x <= 0)
				return 0;
			else
				return 1;
		}
	};

	class tanh_activation : public activation {
	public:
		float_t f(float_t x) const {
			const float_t ep = std::exp(x);
			const float_t em = std::exp(-x);
			return (ep - em) / (ep + em);
		}

		// fast approximation of tanh (improve 2-3% speed in LeNet-5)
		/*float_t f(float_t x) const {
		const float_t x2 = x * x;
		x *= 1.0 + x2 * (0.1653 + x2 * 0.0097);
		return x / std::sqrt(1.0 + x * x);// invsqrt(static_cast<float>(1.0 + x * x));
		}*/

		float_t df(float_t f_x) const { return 1.0 - f_x * f_x; }

	private:
		/*float invsqrt(float x) const {
		float x2 = x * 0.5f;
		long i = *reinterpret_cast<long*>(&x);

		i = 0x5f3759df - (i >> 1);
		x = *reinterpret_cast<float*>(&i);
		x = x * (1.5f - (x2 * x * x));
		return x;
		}*/
	};
} // namespace mlp

#endif //ACTIVATION_H_