//--------------------------------------------------------------------------
// Project: 
// Read GPU temepratures on a CUDA-enabled system.
// Bypass the need for 3rd party libraries.
// Insert into your code as desired. 
//
// Prerequisites: 
// Must have installed the CUDA toolkit.
// Must be running on a UNIX machine
//
// Independent testing info:
// Compile on commandline: nvcc read_temps.cu -o test
// run on commandline: ./test
//
// Author: Jordan Bonilla
// Date  : April 2016
// License: All rights Reserved. See LICENSE.txt
//--------------------------------------------------------------------------

#include <cstdio> // printf
#include <cstring> // memcpy
#include <stdlib.h> // popen, pclose, atoi, fread
#include <cuda_runtime.h> // cudaGetDeviceCount

// Read temperatures of all connected NVIDIA GPUs
void read_temps() 
{
	// Get the number of GPUs on this machine
	int num_devices;
	cudaGetDeviceCount(&num_devices);
	if(num_devices == 0) {
	  printf("No NVIDIA GPUs detected\n");
		return;
	}
	// Read GPU info into buffer "output" using the "nvidia-smi" command
	const unsigned int MAX_BYTES = 10000;
	const unsigned int MAX_DIGITS_PER_INT = 10;
	char output[MAX_BYTES];
	FILE *fp = popen("nvidia-smi &> /dev/null", "r");
	fread(output, sizeof(char), MAX_BYTES, fp);
	pclose(fp);
	// array to hold GPU temperatures
	int * temperatures = new int[num_devices];
	// parse output for temperatures using knowledge of "nvidia-smi" output format
	int i = 0;
	unsigned int num_temps_parsed = 0;
	while(output[i] != '\0') {
		if(output[i] == '%') {
			unsigned int temp_begin = i + 1;
			while(output[i] != 'C') {
				++i;
			}
			unsigned int temp_end = i;
			char this_temperature[MAX_DIGITS_PER_INT];
			memcpy(this_temperature, output + temp_begin, temp_end);
			temperatures[num_temps_parsed] = atoi(this_temperature);
			num_temps_parsed++;
		}
		++i;
	}
	for (int i = 0; i < num_devices; i++) 
	{
	  printf("GPU %d temperature: %d C\n", i, temperatures[i]);
	}

	// Free memory and return
	delete(temperatures);
	return;
}

int main(int argc, char **argv) {
	read_temps();
	return 0;
}
