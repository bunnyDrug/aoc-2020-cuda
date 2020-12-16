//
// Created by Don on 13/12/2020.
//
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
using namespace std;

__global__ void sequential(int *foo, int *bar, int N) {
    for (int i = 0; i <= N; i++) {
        for (int x = 0; x <= N; x++) {
            if (foo[i] + foo[x] == 2020) {
                bar[0] = foo[i];
                bar[1] = foo[x];
                bar[2] = bar[0] * bar[1];
                i = N + 1;
                break;
            }
        }
    }
}

__global__ void babyStride(int *foo, int *bar, int N) {

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index ; i <= N; i+= stride) {
        for (int x = i; x > -1; x--) {
            if (foo[i] + foo[x] == 2020) {
                bar[0] = foo[i];
                bar[1] = foo[x];
                bar[2] = bar[0] * bar[1];
                i = N + 1;
                break;
            }
        }
    }
}


int main() {

    // make some variables
    int *inputData, *sequentialAttemptResults, *babyStrideAttemptResult;

    // Get File line number to create array size.
    std::ifstream file("../input.txt");
    int N = count(istreambuf_iterator<char>(file), istreambuf_iterator<char>(), '\n') + 1;
    file.clear();
    file.seekg(0);

    cudaMallocManaged(&inputData, N * (sizeof(long)));
    cudaMallocManaged(&sequentialAttemptResults, 2 * (sizeof(long)));
    cudaMallocManaged(&babyStrideAttemptResult, 2 * (sizeof(long)));

    // Read file - populate input data array
    string line;
    N = 0;
    while (getline(file, line)) {
        // Output the text from the file
        inputData[N] = stoi(line);
        N++;
    }
    file.close();

    // GPU COMPUTE STARTS HERE:

    //             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
    //  GPU activities:  100.00%  1.4049ms         1  1.4049ms  1.4049ms  1.4049ms  sequential(int*, int*, int)
//    sequential<<<1, 1>>>(inputData, sequentialAttemptResults, N);

    //             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
    //  GPU activities:  100.00%  17.089us         1  17.089us  17.089us  17.089us  babyStride(int*, int*, int)
    babyStride<<<1, 256>>>(inputData, babyStrideAttemptResult, N);

    // Wait for GPU work to finish.
    cudaDeviceSynchronize();

    // ANSWERS for sequential:
    printf("matching value s were %d\n", sequentialAttemptResults[0]);
    printf("matching values were %d\n", sequentialAttemptResults[1]);
    printf("Product = %d\n\n", sequentialAttemptResults[2]);

    // ANSWERS for a basic stride:
    printf("matching value s were %d\n", babyStrideAttemptResult[0]);
    printf("matching values were %d\n", babyStrideAttemptResult[1]);
    printf("Product = %d\n\n", babyStrideAttemptResult[2]);

    cudaFree(inputData);
    cudaFree(sequentialAttemptResults);
    cudaFree(babyStrideAttemptResult);
    return 0;
}
