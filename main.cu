//
// Created by Don on 13/12/2020.
//
#include <iostream>
#include <fstream>
#include <string>

void printProduct(const int *part2ResultsBasicStride, int numberOfLines);

using namespace std;

__global__ void partOneDeviceCodeSequential(int *foo, int *bar, int N) {
    for (int i = 0; i <= N; i++) {
        for (int x = i; x > -1; x--) { // no more duplicate addition.
            if (foo[i] + foo[x] == 2020) {
                bar[0] = foo[i] * foo[x];
                return; // break
            }
        }
    }
}

__global__ void part1DeviceCodeBasicStride(int *foo, int *bar, int N) {

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index ; i <= N; i+= stride) {
        for (int x = i; x > -1; x--) { // no more duplicate compares.
            if (foo[i] + foo[x] == 2020) {
                bar[index] = foo[i] * foo[x];
                return; // break
            }
        }
    }
}


__global__ void part2DeviceCodeBasicStride(int *foo, int *bar, int N) {

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index ; i <= N; i+= stride) {
        for (int x = i; x > -1; x--) { // no more duplicate compares.
            if (foo[i] + foo[x] == 2020) {
                bar[index] = foo[i] * foo[x];
                return; // break
            }
        }
    }
}

__global__ void part2DeviceCodeBasicStridePlus(int *foo, int *bar, int N) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index ; i <= N; i+= stride) {
        for (int x = i; x > -1; x--) { // no more duplicate compares.
            for (int y = x; y > -1; y--) { // no more duplicate compares.
                    if (foo[i] + foo[x] + foo[y] == 2020) {
                        bar[index] = foo[i] * foo[x] * foo[y];
                        return; // break
                    }
            }
        }
    }
}

int main() {
    int *inputData;
    int *part1ResultsSequential;
    int *part1ResultsBasicStride;
    int *part2ResultsBasicStride;


    // Get line numbers from input inputFile.
    ifstream inputFile("../input.txt");
    int numberOfLines = count(istreambuf_iterator<char>(inputFile), istreambuf_iterator<char>(), '\n') + 1;
    // reset inputFile read position
    inputFile.clear();
    inputFile.seekg(0);

    // shared memory allocation for both device and host.
    cudaMallocManaged(&inputData, (sizeof(int))); // input data
    cudaMallocManaged(&part1ResultsSequential, (sizeof(int))); // pt1 (sequential)
    cudaMallocManaged(&part1ResultsBasicStride, (sizeof(int))); // pt1 (stride)
    cudaMallocManaged(&part2ResultsBasicStride, (sizeof(int))); // pt2 (stride)

    // READ FILE
    // Add entries from inputFile as elements to array
    string line;
    numberOfLines = 0;
    while (getline(inputFile, line)) {
        // Output the text from the inputFile
        inputData[numberOfLines] = stoi(line);
        numberOfLines++;
    }
    inputFile.close();


    // GPU COMPUTE STARTS HERE:
    partOneDeviceCodeSequential<<<1, 1>>>(inputData, part1ResultsSequential, numberOfLines);

    part1DeviceCodeBasicStride<<<1, 512>>>(inputData, part1ResultsBasicStride, numberOfLines);

    part2DeviceCodeBasicStridePlus<<<1, 512>>>(inputData, part2ResultsBasicStride, numberOfLines);


    // Wait for GPU work to finish.
    cudaDeviceSynchronize();

    // ANSWERS -->
    printf("Part 1 (sequential device code)\n");
    printProduct(part1ResultsSequential, numberOfLines);

    printf("Part 1 (stride device code)\n");
    printProduct(part1ResultsBasicStride, numberOfLines);

    printf("Part 1 (stride thread safe)\n");
    printProduct(part2ResultsBasicStride, numberOfLines);


    cudaFree(inputData);
    cudaFree(part1ResultsSequential);
    cudaFree(part1ResultsBasicStride);
    cudaFree(part2ResultsBasicStride);
    return 0;
}

void printProduct(const int *results, int N) {
    for (int i = 0; i < N; i++) {
        if (results[i] != 0){
            printf("Product: %d", results[i]);
        }
    }
    printf("\n\n");
}
