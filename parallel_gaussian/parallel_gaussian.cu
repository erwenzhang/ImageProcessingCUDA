#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstring>
#include <algorithm> 
#include "Image.h"
#include "math.h"
#include "utils.h"
#include "timer.h"
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

Image readPPM(const char *filename)
{
    ifstream ifs(filename,ifstream::in);

    Image img;
    try {
        if (ifs.fail()) { throw("Can't open input file"); }
        std::string header;
        int w, h, b;
        ifs >> header;
        if (strcmp(header.c_str(), "P6") != 0) throw("Can't read input file");
        ifs >> w >> h >> b;
        img.w = w; img.h = h;
        img.pixels = new Image::Rgb[w * h]; // this is throw an exception if bad_alloc
        ifs.ignore(256, '\n'); // skip empty lines in necessary until we get to the binary data
        unsigned char pix[3];
        // read each pixel one by one and convert bytes to floats
        for (int i = 0; i < w * h; ++i) {
            ifs.read(reinterpret_cast<char *>(pix), 3);
            img.pixels[i].r = pix[0] / 255.f;
            img.pixels[i].g = pix[1] / 255.f;
            img.pixels[i].b = pix[2] / 255.f;
        }
        ifs.close();
    }
    catch (const char *err) {
        fprintf(stderr, "%s\n", err);
        ifs.close();
    }
    return img;
}

void savePPM(float* ptr, int w, int h, const char *filename)
{

    if (w == 0 || h == 0) { fprintf(stderr, "Can't save an empty image\n"); return; }
    std::ofstream ofs;
    try {
        ofs.open(filename, std::ios::binary); // need to spec. binary mode for Windows users
        if (ofs.fail()) throw("Can't open output file");
        ofs << "P6\n" << w << " " << h << "\n255\n";
        unsigned char r, g, b;
        // loop over each pixel in the image, clamp and convert to byte format
        for (int i = 0; i < w * h; ++i) {
            r = static_cast<unsigned char>(std::min(1.f, ptr[i*3]) * 255);
            g = static_cast<unsigned char>(std::min(1.f, ptr[i*3+1]) * 255);
            b = static_cast<unsigned char>(std::min(1.f, ptr[i*3+2]) * 255);
            ofs << r << g << b;
        }
        ofs.close();
    }
    catch (const char *err) {
        fprintf(stderr, "%s\n", err);
        ofs.close();
    }
}


void createFilter(int filterWidth, float** filter){
    const int blurKernelWidth = filterWidth;
    const float blurKernelSigma = 2.;
    *filter = new float[blurKernelWidth*blurKernelWidth];
    
    float filterSum = 0.f;
    
    for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
        for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
            float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
            (*filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
            filterSum += filterValue;
        }
    }
    
    float normalizationFactor = 1.f / filterSum;
    
    for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
        for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
            (*filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
        }
    }
}

void gaussianBlur(const Image* inputImage, Image* outputImage, int numRows, int numCols, const float* const filter, const int filterWidth){
    for(int i=0;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            float result_r = 0.f;
            float result_g = 0.f;
            float result_b = 0.f;
            for(int filterRow = -filterWidth/2; filterRow <= filterWidth/2; ++filterRow){
                for(int filterCol = -filterWidth/2; filterCol <= filterWidth/2; ++filterCol){
                    int imageR = min(max(i+filterRow,0),static_cast<int> (numRows-1));
                    int imageC = min(max(j+filterCol,0),static_cast<int> (numCols-1));
                    float filterVal = filter[(filterRow+filterWidth/2)*filterWidth + filterCol + filterWidth/2];
                    float imageVal_r = static_cast<float> (inputImage->pixels[imageR*numCols + imageC].r);
                    float imageVal_g = static_cast<float> (inputImage->pixels[imageR*numCols + imageC].g);
                    float imageVal_b = static_cast<float> (inputImage->pixels[imageR*numCols + imageC].b);   
                    result_r += imageVal_r*filterVal;
                    result_g += imageVal_g*filterVal;
                    result_b += imageVal_b*filterVal;
                    
                }
            }
            outputImage->pixels[i*numRows+j].r = result_r;
            outputImage->pixels[i*numRows+j].g = result_g;
            outputImage->pixels[i*numRows+j].b = result_b;
        }
    }
}

__global__ void test_kernel(float* outputChannel, const float* inputChannel, int numRows, int numCols, const float * d_in, const int filterWidth)
{
    const int2 thread2DPos = make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    const int thread1DPos = thread2DPos.y * numCols + thread2DPos.x;
    if(thread1DPos < 81)
   	 printf("block %d, thread %d, value is %f\n", blockIdx.x, thread1DPos, d_in[thread1DPos]);
    if(thread2DPos.x >= numCols || thread2DPos.y >= numRows) return;
    float result = 0.f;
    for(int filterRow = -filterWidth/2; filterRow <= filterWidth/2; ++filterRow){
        for(int filterCol = -filterWidth/2; filterCol <= filterWidth/2; ++filterCol){
            // if(thread1DPos<1000)
            //     printf("filterRow: %d, filterCol: %d\n", filterRow, filterCol);
            //     __syncthreads();
            int imageR = min(max(thread2DPos.y+filterRow,0),static_cast<int> (numRows-1));
            int imageC = min(max(thread2DPos.x+filterCol,0),static_cast<int> (numCols-1));
                // if(thread1DPos<1000)
                // printf("imageR: %d, imageC: %d\n", imageR, imageC);
                // __syncthreads();
            float imageVal = static_cast<float> (inputChannel[imageR*numCols + imageC]);
               result += imageVal*0.01;
	}
    }
    outputChannel[thread1DPos] = result;
}

__global__ void gaussianBlur(float* outputChannel, const float* inputChannel, int numRows, int numCols, const float* filter, const int filterWidth){
    const int2 thread2DPos = make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    const int thread1DPos = thread2DPos.y * numCols + thread2DPos.x;
    // printf("hello 1\n");
    // printf("*filter: %f\n", *filter);
    if( thread1DPos < 100){
    printf("filter[0]: %f\n", filter[0]);
    }
    // printf("hello 2\n");
    // __syncthreads();
    if(thread2DPos.x >= numCols || thread2DPos.y >= numRows) return;

    float result = 0.f;
    for(int filterRow = -filterWidth/2; filterRow <= filterWidth/2; ++filterRow){
        for(int filterCol = -filterWidth/2; filterCol <= filterWidth/2; ++filterCol){
            // if(thread1DPos<1000)
            //     printf("filterRow: %d, filterCol: %d\n", filterRow, filterCol);
            //     __syncthreads();
            int imageR = min(max(thread2DPos.y+filterRow,0),static_cast<int> (numRows-1));
            int imageC = min(max(thread2DPos.x+filterCol,0),static_cast<int> (numCols-1));
                // if(thread1DPos<1000)
                // printf("imageR: %d, imageC: %d\n", imageR, imageC);
                // __syncthreads();
            float imageVal = static_cast<float> (inputChannel[imageR*numCols + imageC]);
                if(thread1DPos<1000){
                // printf("imageVal %f\n", imageVal);
                printf("filterPos %d\n", (filterRow+filterWidth/2)*filterWidth + filterCol + filterWidth/2);
                }
                __syncthreads();
            // float filterVal = filter[static_cast<int>((filterRow+filterWidth/2)*filterWidth + filterCol + filterWidth/2)];
            // float filterVal = filter[0];
                // if(thread1DPos<1000)
                // printf("filterVal %f\n", filterVal);
                // __syncthreads();                     
            // result += imageVal*filterVal;
                result += imageVal*0.01;
            if(thread1DPos<10000)
            printf("result %f\n", result);
            printf("hello 3\n");
            printf("thread1DPos %d, value is %f\n", thread1DPos, result);
        }
    }
                                                              
    if(thread1DPos<100000)
    printf("thread1DPos %d, value is %f\n", thread1DPos, result);
    outputChannel[thread1DPos] = result;
}

int main(int argc, const char * argv[]) {
    // insert code here...
    Image inputImage = readPPM("lena.ppm");
    int w = inputImage.w, h = inputImage.h;

    float* filter;
    int filterWidth = 9;
    createFilter(filterWidth, &filter);

    Image outputImage;
    outputImage.w = w; outputImage.h = h;
    outputImage.pixels = new Image::Rgb[w * h];  


    float *input_r, *input_g, *input_b;
    float *output_r, *output_g, *output_b;
    float* gpu_filter;
    cudaMalloc((void **)&input_r, w*h * sizeof(float));
    cudaMalloc((void **)&input_g, w*h * sizeof(float));
    cudaMalloc((void **)&input_b, w*h * sizeof(float));
    cudaMalloc((void **)&output_r, w*h * sizeof(float));
    cudaMalloc((void **)&output_g, w*h * sizeof(float));
    cudaMalloc((void **)&output_b, w*h * sizeof(float));
    cudaMalloc((void **)&gpu_filter, filterWidth*filterWidth * sizeof(float));


    float* ptr = &(inputImage.pixels[0].r);

    cudaMemcpy2D(input_r, sizeof(float), ptr, 3*sizeof(float), sizeof(float), w*h, cudaMemcpyHostToDevice);
    cudaMemcpy2D(input_g, sizeof(float), ptr+1, 3*sizeof(float), sizeof(float), w*h, cudaMemcpyHostToDevice);
    cudaMemcpy2D(input_b, sizeof(float), ptr+2, 3*sizeof(float), sizeof(float), w*h, cudaMemcpyHostToDevice);
    printf("filter at 0 is %f\n", filter[0]);
    cudaMemcpy(gpu_filter, filter, filterWidth*filterWidth * sizeof(float), cudaMemcpyHostToDevice);

    // int blocks = 256;
    // int threads = w*h/blocks;

    const dim3 blockSize(32,32,1);
    const dim3 gridSize(w/32,h/32,1);
    test_kernel<<<gridSize, blockSize>>>(output_r, input_r, h, w, gpu_filter, filterWidth);   
    //gaussianBlur<<<gridSize, blockSize>>>(output_r, input_r, h, w, gpu_filter, filterWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    test_kernel<<<gridSize, blockSize>>>(output_g, input_g, h, w, gpu_filter, filterWidth);
    cudaDeviceSynchronize();
    test_kernel<<<gridSize, blockSize>>>(output_b, input_b, h, w, gpu_filter, filterWidth);
    cudaDeviceSynchronize();

    cudaMemcpy2D(ptr, 3*sizeof(float), output_r, sizeof(float), sizeof(float),w*h, cudaMemcpyHostToDevice);
    cudaMemcpy2D(ptr+1, 3*sizeof(float), output_b, sizeof(float), sizeof(float),w*h, cudaMemcpyHostToDevice);
    cudaMemcpy2D(ptr+2, 3*sizeof(float), output_g, sizeof(float), sizeof(float),w*h, cudaMemcpyHostToDevice);
    printf("r0 is %f, b1 is %f, g1 is %f\n", ptr[0], ptr[1], ptr[2]);

    cudaFree(input_r);
    cudaFree(input_g);
    cudaFree(input_b);

    cudaFree(output_r);
    cudaFree(output_g);
    cudaFree(output_b);

    printf("finished here\n");
    cudaFree(gpu_filter);



    // gaussianBlur(&inputImage, &outputImage, h, w, filter, filterWidth);
    savePPM(ptr, w, h,"result.ppm");
    printf("output finished\n");

    return 0;
}
