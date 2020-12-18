//
// Created by Don on 13/12/2020.
//
#include <iostream>
#include <fstream>
#include <string>

void printProduct(const int *part2ResultsBasicStride);

using namespace std;

__global__ void part1DeviceCodeBasicStride(int *foo, int *bar, int N) {

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index ; i < N; i+= stride) {
        for (int x = i; x > -1; x--) { // no more duplicate compares.
            if (foo[i] + foo[x] == 2020) {
                bar[0] = foo[i] * foo[x];
                return; // break
            }
        }
    }
}


__global__ void part2DeviceCodeBasicStride(int *foo, int *bar, int N) {

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index ; i < N; i+= stride) {
        for (int x = i; x > -1; x--) { // no more duplicate compares.
            for (int y = x; y > -1; y--) { // no more duplicate compares.
                    if (foo[i] + foo[x] + foo[y] == 2020) {
                        bar[0] = foo[i] * foo[x] * foo[y];
                        return; // break
                    }
            }
        }
    }
}

int main() {
    int *input;
    int *part1ResultsBasicStride;
    int *part2ResultsBasicStride;

    // shared memory allocation for both device and host.
    cudaMallocManaged(&input, (sizeof(int))); // input data
    cudaMallocManaged(&part1ResultsBasicStride, (sizeof(int))); // pt1 (stride)
    cudaMallocManaged(&part2ResultsBasicStride, (sizeof(int))); // pt2 (stride)

    // READ FILE
    ifstream inputFile("../input.txt");
    string line;
    int N = 0;
    while (getline(inputFile, line)) {
        // Output the text from the inputFile
        input[N] = stoi(line);
        N++;
    }
    inputFile.close();


    // GPU COMPUTE STARTS HERE:
    part1DeviceCodeBasicStride<<<1, 512>>>(input, part1ResultsBasicStride, N);
    part2DeviceCodeBasicStride<<<1, 512>>>(input, part2ResultsBasicStride, N);

    // Wait for GPU work to finish.
    cudaDeviceSynchronize();

    // ANSWERS -->
    printf("Part 1\n");
    printProduct(part1ResultsBasicStride);

    printf("Part 2\n");
    printProduct(part2ResultsBasicStride);

    cudaFree(input);
    cudaFree(part1ResultsBasicStride);
    cudaFree(part2ResultsBasicStride);
    return 0;
}

void printProduct(const int *results) {
    printf("Product: %d\n\n", results[0]);
}
