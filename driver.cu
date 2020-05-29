// Berat Postalcioglu
/*OUTPUT

	blocksPerGrid   threadsPerBlock   time to generate
	-------------   ---------------   ----------------
		 157           256              0.04400000 ms.
		  79           512              0.05434880 ms.
		  40          1024              0.09233920 ms.
	   40000             1              0.00174080 ms.
	   20000             1              0.00174080 ms.
	   10000             1              0.00179200 ms.
		5000             1              0.00189440 ms.
		1000             1              0.07427360 ms.
		   2             1             27.85148430 ms.
		   1             1             54.44231415 ms.

*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <cmath>
#include <iostream>
#include <cstdio>

using namespace std;

const int ArrSize = 40000;


__global__ void diffGPU(double *a, double *b, double *c, int size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < size)
	{
		c[tid] = a[tid] - b[tid];
		tid += blockDim.x * gridDim.x;
	}
}


void generateArray(double *arr, double size)
{
	for (int i = 0; i < size; i++)
	{
		arr[i] = rand() % 100 + 1;
	}
}

double sumArray(double *data, int count)
{
	double result = 0;
	for (int i = 0; i < count; i++)
	{
		result += data[i];
	}
	return result;
}

double sumArrayDiff(double *a, double *b, int count)
{
	double *c=new double[count];
	for (int i = 0; i < count; i++)
	{
		c[i] = a[i] - b[i];
	}

	double result = sumArray(c, count);
	delete[] c;

	return result;
}

int* diffCPU(int *v1, int *v2)
{
	int res[ArrSize];
	for (int i = 0; i < ArrSize; i++)
	{
		res[i] = v1[i] - v2[i];
	}
	return res;
}

void displayArray(int *arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		std::cout << arr[i] << " ";
	}
	std::cout << std::endl;
}

float duration(double *devA, double *devB, double *devC, int blocksPerGrid, int threadsPerBlock, double resultCPU)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	// gpu work
	diffGPU <<<threadsPerBlock, blocksPerGrid >>> (devA, devB, devC, ArrSize);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	double *resFromGpu = new double[ArrSize];
	cudaMemcpy(resFromGpu, devC, sizeof(double)*ArrSize, cudaMemcpyDeviceToHost);
	double sum = sumArray(resFromGpu, ArrSize);	delete[] resFromGpu;	if (sum != resultCPU)	{		cout << "Results from CPU and GPU are not same... Exiting..." << endl;		exit(1);	}	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	return elapsedTime;
}

float averageDuration(double *devA, double *devB, double *devC,
	int blocksPerGrid, int threadsPerBlock, double resultCPU, int repetition)
{
	double totalElapsedTime = 0;
	for (int i = 0; i < repetition; i++)
	{
		totalElapsedTime += duration(devA, devB, devC, blocksPerGrid, threadsPerBlock, resultCPU);
	}
	return (totalElapsedTime / repetition);
}

void prepareTable(double *devA, double *devB, double *devC, int blocksPerGrid, int threadsPerBlock, double resultCPU, int repetition) {

	double ave = averageDuration(devA, devB, devC, blocksPerGrid, threadsPerBlock, resultCPU, repetition);
	printf("%8d", blocksPerGrid);
	printf("%14d", threadsPerBlock);
	printf("%24.8f", ave);
	printf(" ms.\n");
}

int main()
{
	srand(time(NULL));

	// cpu
	double a[ArrSize], b[ArrSize], c[ArrSize];
	generateArray(a, ArrSize);
	generateArray(b, ArrSize);
	double resultCPU = sumArrayDiff(a, b, ArrSize);

	// gpu
	unsigned int totalBytes = ArrSize * sizeof(double);
	double *devA, *devB, *devC;
	cudaMalloc(&devA, totalBytes);
	cudaMalloc(&devB, totalBytes);
	cudaMalloc(&devC, totalBytes);

	cudaMemcpy(devA, a, totalBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(devB, b, totalBytes, cudaMemcpyHostToDevice);

	cout << "blocksPerGrid" << "   " << "threadsPerBlock" << "   " << "time to generate" << endl;
	cout << "-------------" << "   " << "---------------" << "   " << "----------------" << endl;
	prepareTable(devA, devB, devC, 157, 256, resultCPU, 1);
	prepareTable(devA, devB, devC, 79, 512, resultCPU, 20);
	prepareTable(devA, devB, devC, 40, 1024, resultCPU, 20);
	prepareTable(devA, devB, devC, 40000, 1, resultCPU, 20);
	prepareTable(devA, devB, devC, 20000, 1, resultCPU, 20);
	prepareTable(devA, devB, devC, 10000, 1, resultCPU, 20);
	prepareTable(devA, devB, devC, 5000, 1, resultCPU, 20);
	prepareTable(devA, devB, devC, 1000, 1, resultCPU, 20);
	prepareTable(devA, devB, devC, 2, 1, resultCPU, 20);
	prepareTable(devA, devB, devC, 1, 1, resultCPU, 20);

	return 0;
}