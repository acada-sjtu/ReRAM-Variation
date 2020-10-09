#ifndef FAULT_H_
#define FAULT_H_
#pragma once
#include "util.h"
#include <cassert>


boost::mt19937 gen(0);


namespace mlp{
	class Fault
	{
	public:
		
		
		void generateVariation(double sigma)
		{
			assert(row != 0 && column != 0);
			/*************************************   set normal variation   *********************************************************/
			boost::mt19937 gen(time(NULL));
			boost::normal_distribution<>dist1(0, sigma);
			boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >die1(gen, dist1);

			for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < column; j++)
				{
					fault_array[i][j].type = 2;

					fault_array[i][j].value = die1();
				}

			}

		}	
		
		void generateLogVariation(double sigma)
		{
			assert(row != 0 && column != 0);
			/*************************************   set log normal variation   *********************************************************/
		//	boost::mt19937 gen(time(NULL));
			boost::normal_distribution<>dist1(0, sigma);
			boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >die1(gen, dist1);

			for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < column; j++)
				{
					fault_array[i][j].type = 2;

					fault_array[i][j].value = 1.0-exp(die1());
				}
			}

		}	
		
		
		void generateSA0(double sa0_fault_ratio, int sa0_faulty_columns)
		{
			for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < column; j++)
				{
					fault_array[i][j].type = 2;
					fault_array[i][j].value = 0;
				}
			}
			/*************************************   set sa0   *********************************************************/
			//set faulty_col
			int sum0 = 0;
			int *fault_col = new int[sa0_faulty_columns];
			boost::mt19937 gen(time(NULL));
			boost::uniform_int<>dist2(0, column - 1);
			boost::variate_generator<boost::mt19937&, boost::uniform_int<> >die2(gen, dist2);
			for (int i = 0; i < sa0_faulty_columns; ++i)
			{
				fault_col[i] = die2();
			}

			for (int i = 0; i < sa0_faulty_columns; ++i)
			{
				boost::uniform_real<>dist3(0.5, 1);
				boost::variate_generator<boost::mt19937&, boost::uniform_real<> >die3(gen, dist3);
				double ratio = die3();
				for (int j = 0; j < row; j++){

					boost::bernoulli_distribution<>dist4(ratio); //P(true)
					boost::variate_generator<boost::mt19937&, boost::bernoulli_distribution<> >die4(gen, dist4);
					if (die4() == 1) 	
					{ 
						boost::bernoulli_distribution<>dist6(0.5); //P(true)
						boost::variate_generator<boost::mt19937&, boost::bernoulli_distribution<> >die6(gen, dist6);
						if(die6() == 1){fault_array[j][fault_col[i]].type = 0; }
						else {fault_array[j][fault_col[i]].type = 3;}
						sum0++; 
					}
				}
			}
			//set random fault

			double d = (sa0_fault_ratio - double(sum0) / row / column) * 2 / row;
			for (int i = 0; i < row; ++i)
			{
				double ratio = d*i;
				for (int j = 0; j < column; j++){

					boost::bernoulli_distribution<>dist5(ratio); //P(true)
					boost::variate_generator<boost::mt19937&, boost::bernoulli_distribution<> >die5(gen, dist5);
					if (die5() == 1) 	
					{
						boost::bernoulli_distribution<>dist7(0.5); //P(true)
						boost::variate_generator<boost::mt19937&, boost::bernoulli_distribution<> >die7(gen, dist7);
						if(die7() == 1){fault_array[i][j].type = 0; }
						else {fault_array[i][j].type = 3;}
					}
				}
			}
			delete[] fault_col;
		}
		
		
		void generateSA1(double sa1_fault_ratio, int sa1_faulty_rows, int sa1_bad_block)
		{
			/*************************************   set sa1   *********************************************************/
			//set bad block
			boost::mt19937 gen(time(NULL));
			int sum1 = 0;
			for (int i = 0; i < sa1_bad_block; i++)
			{
				boost::uniform_real<>dist6(0.5, 0.75);
				boost::variate_generator<boost::mt19937&, boost::uniform_real<> >die6(gen, dist6);
				double ratio = die6();
				for (int j = 0; j < row; j++){
					boost::bernoulli_distribution<>dist7(ratio); //P(true)
					boost::variate_generator<boost::mt19937&, boost::bernoulli_distribution<> >die7(gen, dist7);
					if (die7() == 1) 	{ fault_array[j][i].type = 1; sum1++; }
				}
			}
			//set faulty_row
			int *fault_row = new int[sa1_faulty_rows];
			for (int i = 0; i < sa1_faulty_rows; ++i)
			{
				boost::uniform_int<>dist8(0, row - 1);
				boost::variate_generator<boost::mt19937&, boost::uniform_int<> >die8(gen, dist8);
				fault_row[i] = die8();
			}

			for (int i = 0; i < sa1_faulty_rows; ++i)
			{
				boost::uniform_real<>dist9(0.5, 1);
				boost::variate_generator<boost::mt19937&, boost::uniform_real<> >die9(gen, dist9);
				double ratio = die9();
				for (int j = 0; j < column; j++){
					boost::bernoulli_distribution<>dist10(ratio); //P(true)
					boost::variate_generator<boost::mt19937&, boost::bernoulli_distribution<> >die10(gen, dist10);
					if (die10() == 1) 	{ fault_array[fault_row[i]][j].type = 1; sum1++; }
				}
			}
			//set random fault
			double d = (sa1_fault_ratio - double(sum1) / row / column) * 2 / column;
			for (int j = 0; j < column; j++)
			{
				double ratio = d*j;
				for (int i = 0; i < row; i++){
					boost::bernoulli_distribution<>dist11(ratio); //P(true)
					boost::variate_generator<boost::mt19937&, boost::bernoulli_distribution<> >die11(gen, dist11);
					if (die11() == 1) 	fault_array[i][j].type = 1;
				}
			}
			delete[] fault_row;

		}
		
		void resizeFault(int row_size, int column_size)
		{
			column = column_size;
			row = row_size;
			fault_array.resize(row_size);
			for (int i = 0; i < row_size; i++)
			{
				fault_array[i].resize(column_size);
			}
		}

		float getFaultValue( int row_index, int col_index)
		{
			assert(row_index < row&&col_index < column);
			return fault_array[row_index][col_index].value;
		}
		
		int	getFaultType (int row_index, int col_index)
		{
			assert(row_index < row&&col_index < column);
			return fault_array[row_index][col_index].type;
		}

		void setFaultValue(int row_index, int col_index, double v)
		{
			assert(row_index < row&&col_index < column);
			fault_array[row_index][col_index].value = v;
		}
		
		void setFaultType(int row_index, int col_index, int v)
		{
			assert(row_index < row&&col_index < column);
			fault_array[row_index][col_index].type = v;
		}

		Fault(){}

		Fault(int row_size, int column_size)
		{
			column = column_size;
			row = row_size;
			fault_array.resize(row_size);
			for (int i = 0; i < row_size; i++)
			{
				fault_array[i].resize(column_size);
			}
			//column = 1024;
			//row = 1024;
			//sigma = 0.07;   //variation
			//sa0_fault_ratio = 0.0175;
			//sa1_fault_ratio = 0.0904;
			//sa0_faulty_columns = 9;
			//sa1_faulty_rows = 3;
			//sa1_bad_block = 128;
			//fault_array(column, row, sigma, sa0_fault_ratio, sa1_fault_ratio, sa0_faulty_columns, sa1_faulty_rows, sa1_bad_block);
		};


	private:
		int column=0;
		int row=0;
		vec2d_fault fault_array;
		//double sigma;   //variation
		//double sa0_fault_ratio;
		//double sa1_fault_ratio;
		//int sa0_faulty_columns;
		//int sa1_faulty_rows;
		//int sa1_bad_block;
	};
}



#endif