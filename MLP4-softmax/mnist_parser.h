#ifndef MNIST_PARSER_H_
#define MNIST_PARSER_H_

#pragma once

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "util.h"

namespace mlp{
#define LOAD_MNIST_TEST( data, labels )	\
	_load("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", data, labels);

#define LOAD_MNIST_TRAIN( data, labels ) \
	_load("train-images.idx3-ubyte", "train-labels.idx1-ubyte" ,data, labels);

	std::uint32_t swapEndien_32(std::uint32_t);
	void _load(std::string fimage, std::string flabel, vec2d_t& data, vec_t& labels){
		std::ifstream in;
		in.open(fimage.c_str(), std::ifstream::binary);
		if (!in.is_open()){
			std::cout << "file opened failed." << std::endl;
		}
		std::uint32_t magic = 0;
		std::uint32_t number = 0;
		std::uint32_t rows = 0;
		std::uint32_t cols = 0;
		in.read((char*)&magic, sizeof(uint32_t));
		in.read((char*)&number, sizeof(uint32_t));
		in.read((char*)&rows, sizeof(uint32_t));
		in.read((char*)&cols, sizeof(uint32_t));
		assert(swapEndien_32(magic) == 2051);
		std::cout << "amount:" << swapEndien_32(number) << std::endl;
		assert(swapEndien_32(rows) == 28);
		assert(swapEndien_32(cols) == 28);
		uint8_t pixel = 0;
		vec_t sample;
		size_t index = 0;
		while (!in.eof()){
			in.read((char*)&pixel, sizeof(uint8_t));
			index++;
			sample.push_back(((float_t)pixel)/256);
			if (index % (28 * 28) == 0){
				data.push_back(sample);
				sample.clear();
			}
		}
		in.close();

		assert(data.size() == swapEndien_32(number));

		//label
		in.open(flabel.c_str(), std::ifstream::binary);
		if (!in.is_open()){
			std::cout << "failed opened label file";
		}
		in.read((char*)&magic, sizeof(uint32_t));
		in.read((char*)&number, sizeof(uint32_t));
		assert(2049 == swapEndien_32(magic));
		uint8_t label;
		while (!in.eof())
		{
			in.read((char*)&label, sizeof(uint8_t));
			//std::cout << (float_t)label << std::endl;
			labels.push_back(label);
		}
		in.close();
	}

	// reverse endien for uint32_t
	std::uint32_t swapEndien_32(std::uint32_t value){
		return ((value & 0x000000FF) << 24) |
			((value & 0x0000FF00) << 8) |
			((value & 0x00FF0000) >> 8) |
			((value & 0xFF000000) >> 24);
	}
} // namespace mlp

#endif
