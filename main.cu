//
// Created by Don on 13/12/2020.
//
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

//__global__ void dothings(int *foo, int *bar, int N) {
//    for (int i = 0; i <= N; i++) {
//        for (int x = 0; x <= N; x++) {
//            if (foo[i] + foo[x] == 2020) {
//                bar[0] = foo[i];
//                bar[1] = foo[x];
//                bar[2] = bar[0] * bar[1];
//                i = N + 1;
//                break;
//            }
//        }
//    }
//}

__global__ void stridy(int *foo, int *bar, int N) {

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index ; i <= N; i+= stride) {
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


int main() {

    // make some variables
    int *inputData, *resultData;
    // Create a text string, which is used to output the text file
    string line;


    std::ifstream file("../input.txt");
    int N = count(istreambuf_iterator<char>(file), istreambuf_iterator<char>(), '\n') + 1;
    file.clear();
    file.seekg(0);

    cudaMallocManaged(&inputData, N * (sizeof(long)));
    cudaMallocManaged(&resultData, 2 * (sizeof(long)));


    N = 0;
    // Use a while loop together with the getline() function to read the file line by line
    while (getline(file, line)) {
        // Output the text from the file
        inputData[N] = stoi(line);
        N++;
    }
    file.close();



//    dothings<<<1, 1>>>(inputData, resultData, N);
    //             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
    //  GPU activities:  100.00%  1.4049ms         1  1.4049ms  1.4049ms  1.4049ms  dothings(int*, int*, int)

    stridy<<<1, 256>>>(inputData, resultData, N);
    //            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
    // GPU activities:  100.00%  20.032us         1  20.032us  20.032us  20.032us  stridy(int*, int*, int)


    cudaDeviceSynchronize();

    printf("matching value s were %d\n", resultData[0]);
    printf("matching values were %d\n", resultData[1]);
    printf("Product = %d\n", resultData[2]);


    cudaFree(inputData);
    cudaFree(resultData);
    return 0;
}
