#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <string.h>
#include <stdio.h>
#include <cassert>
#include <time.h>
#include "boost/random.hpp"

using namespace std;

//56909194



float variation[145578];

boost::mt19937 gen(time(NULL));

void generateLogVariation(double sigma)
	{
		
		/*************************************   set log normal variation   *********************************************************/
		
		boost::normal_distribution<>dist1(0, sigma);
		boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >die1(gen, dist1);

		for (int i = 0; i < 145578; i++)
		{
			

				variation[i] = exp(die1());
		}

	}	

void fout_fault(std::ofstream &ofile)
	{
		
			
			for (int i = 0; i < 145578; i++){
				
					ofile << variation[i] << "\n";
			
		
		}
	}

int main()
{
	for (int i = 0; i < 10;i++)
	{
		std::stringstream ss;
		ss << i << ".txt";
		std::ofstream fault_file(ss.str());
		generateLogVariation(1.1);		
		fout_fault(fault_file);
		fault_file.close();
	}

	return 0;
}
