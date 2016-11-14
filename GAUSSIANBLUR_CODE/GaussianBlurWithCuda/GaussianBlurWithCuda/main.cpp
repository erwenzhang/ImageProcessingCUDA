//
//  main.cpp
//  GaussianBlurWithCuda
//
//  Created by Wenwen on 11/10/16.
//  Copyright © 2016 wenwen. All rights reserved.
//
using namespace std;
#include <iostream>
#include <string>
#include <stdio.h>
#include <fstream>
#include "Image.h"
#include "utils.h"
#include "timer.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gaussianBlur(const unsigned char* inputChannel, unsigned char* const outputChannel, int numRows, int numCols, const float* const filter, const int filterWidth){
    const int2 thread2DPos = make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    const int thread1DPos = thread2DPos.y * numCols + thread2DPos.x;
    
    if(thread2DPos.x >= numCols ||thread2DPos.y >= numRows) return;
    
    float result = 0.f;
    for(int filterRow = -filterWidth/2; filterRow <= filterWidth/2; ++filterRow){
        for(int filterCol = -filterWidth/2; filterCol <= filterWidth/2; ++filterCol){
            int imageR = min(max(thread2DPos.y+filterRow,0),static_cast<int> (numRows-1));
            int imageC = min(max(thread2DPos.x+filterCol,0),static_cast<int> (numCols-1));
            float imageVal = static_cast<float> (inputChannel[imageR*numCols + imageC]);
            float filterVal = filter[(filterRow+filterWidth/2)*filterWidth + filterCol + filterWidth/2];
                                                              
            result += imageVal*filterVal;
        }
    }
                                                              
    outputChannel[thread1DPos] = result;
}
                                                              
                                                              
__global__ void separateChannels(const Image::Rgb* inputImageDevice, int numRows, int numCols, unsigned char* const redChannel,unsigned char* const greenChannel, unsigned char* const blueChannel){
    const int2 thread2DPos = make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    const int thread1DPos = thread2DPos.y * numCols + thread2DPos.x;
    printf("hello from Thread: %d, %d, %d, %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    printf("hello from Thread2: %d, %d, %d\n", thread2DPos.x, thread2DPos.y, thread1DPos);  
      __syncthreads(); 
    if(thread2DPos.x >= numCols ||thread2DPos.y >= numRows)     return;
    redChannel[thread1DPos] = inputImageDevice[thread1DPos].r;
    greenChannel[thread1DPos] = inputImageDevice[thread1DPos].g;
    blueChannel[thread1DPos] = inputImageDevice[thread1DPos].b;
}
                                                              
__global__ void combineChannels(const unsigned char* const redChannel,const unsigned char* const greenChannel, const unsigned char* const blueChannel, Image::Rgb* const outputImageDevice, int numRows,int numCols,Image::Rgb tmpPixel){
    const int2 thread2DPos = make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    const int thread1DPos = thread2DPos.y * numCols + thread2DPos.x;
                                                              
    if(thread2DPos.x >= numCols ||thread2DPos.y >= numRows) return;
                                                              
    unsigned char red = redChannel[thread1DPos];
    unsigned char green = greenChannel[thread1DPos];
    unsigned char blue = blueChannel[thread1DPos];
    // Image::Rgb pixel;
    tmpPixel.r = red; 
    tmpPixel.g = green;
    tmpPixel.b = blue;
    outputImageDevice[thread1DPos] = tmpPixel;
}

Image imageInput;

// unsigned char *red,*green,*blue;
float *filterDevice;
Image::Rgb* inputImageDMarker;
Image::Rgb* outputImageDMarker;
float* filterHostMarker;

Image readPPM(const char *filename)
{
    ifstream ifs(filename,ifstream::in);
//    ifs.open(filename, std::ios::binary); // need to spec. binary mode for Windows users

    Image img;
    try {
        if (ifs.fail()) { throw("Can't open input file"); }
        std::string header;
        int w, h, b;
        ifs >> header;
        if (strcmp(header.c_str(), "P6") != 0) throw("Can't read input file");
        ifs >> w >> h >> b;
        img.w = w; img.h = h;
        img.rows = h;
        img.cols = w;

        // printf("img.w = %d\n",img.cols);
        // printf("img.h = %d\n",img.rows);

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

void savePPM(Image::Rgb** outputImageHost, const char *filename)
{

    // if (img.w == 0 || img.h == 0) { fprintf(stderr, "Can't save an empty image\n"); return; }
    std::ofstream ofs;
    try {
        ofs.open(filename, std::ios::binary); // need to spec. binary mode for Windows users
        if (ofs.fail()) throw("Can't open output file");
        ofs << "P6\n" << imageInput.w << " " << imageInput.h << "\n255\n";
        unsigned char r, g, b;
        // loop over each pixel in the image, clamp and convert to byte format
        for (int i = 0; i < imageInput.w * imageInput.h; ++i) {
            r = static_cast<unsigned char>(std::min(1.f, outputImageHost[i]->r) * 255);
            g = static_cast<unsigned char>(std::min(1.f, outputImageHost[i]->g) * 255);
            b = static_cast<unsigned char>(std::min(1.f, outputImageHost[i]->b) * 255);
            ofs << r << g << b;
        }
        ofs.close();
    }
    catch (const char *err) {
        fprintf(stderr, "%s\n", err);
        ofs.close();
    }
}

void preProcess(const char *filename, Image::Rgb** inputImageHost){
    
//    checkCudaErrors(cudaFree(0));
    printf("entering Preprocessing\n"); 
    imageInput = readPPM("lena.ppm");
    printf("image w=%d\n",imageInput.cols);
    printf("image h=%d\n",imageInput.rows);
    *inputImageHost = (Image::Rgb *)imageInput.pixels;
    printf("pixel1: %d pixel2: %d pixel3: %d pixel4: %d", inputImageHost[0].r,inputImageHost[1].g,inputImageHost[2].b,inputImageHost[3].r);
}
void postProcess(const char* output_file,Image::Rgb** outputImageHost ) { 
    savePPM(outputImageHost, output_file);
}

void createFilterAndCopyToGPU(int* filterWidth, float** filterHost){
    const int blurKernelWidth = 9;
    const float blurKernelSigma = 2.;
    *filterWidth = blurKernelWidth;
    *filterHost = new float[blurKernelWidth*blurKernelWidth];
    filterHostMarker = *filterHost;
    
    float filterSum = 0.f;
    
    for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
        for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
            float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
            (*filterHost)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
            filterSum += filterValue;
        }
    }
    
    float normalizationFactor = 1.f / filterSum;
    
    for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
        for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
            (*filterHost)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
        }
    }

    checkCudaErrors(cudaMalloc((void**)&filterDevice,sizeof(float)*(*filterWidth)*(*filterWidth)));
    checkCudaErrors(cudaMemcpy(filterDevice,filterHost,sizeof(float)*(*filterWidth)*(*filterWidth),cudaMemcpyHostToDevice));
    
}

void allocateMemoryAndCopyToGPU(Image::Rgb* inputImageHost, 
                                Image::Rgb* inputImageDevice, 
                                Image::Rgb* outputImageDevice, 
                                unsigned char * redBlurredDevice,
                                unsigned char * greenBlurredDevice, 
                                unsigned char* blueBlurredDevice,
                                unsigned char* red,
                                unsigned char* green,
                                unsigned char* blue,
    ){
    const size_t numPixels = imageInput.rows*imageInput.cols;
    printf("numPixels =%d\n",numPixels );
    checkCudaErrors(cudaMalloc((void**)&red,sizeof(unsigned char)*numPixels));
    checkCudaErrors(cudaMalloc((void**)&green,sizeof(unsigned char)*numPixels));
    checkCudaErrors(cudaMalloc((void**)&blue,sizeof(unsigned char)*numPixels));

    // checkCudaErrors(cudaMalloc((void**)&redBlurredDevice,    sizeof(unsigned char) * numPixels));
    // checkCudaErrors(cudaMalloc((void**)&greenBlurredDevice,  sizeof(unsigned char) * numPixels));
    // checkCudaErrors(cudaMalloc((void**)&blueBlurredDevice,   sizeof(unsigned char) * numPixels));
    // checkCudaErrors(cudaMemset(redBlurredDevice,   0, sizeof(unsigned char) * numPixels));
    // checkCudaErrors(cudaMemset(greenBlurredDevice, 0, sizeof(unsigned char) * numPixels));
    // checkCudaErrors(cudaMemset(blueBlurredDevice,  0, sizeof(unsigned char) * numPixels));

    checkCudaErrors(cudaMalloc((void**)&inputImageDevice,sizeof(Image::Rgb)*numPixels));
    // checkCudaErrors(cudaMalloc((void**)&outputImageDevice,sizeof(Image::Rgb)*numPixels));
    // checkCudaErrors(cudaMemset(outputImageDevice,0,sizeof(Image::Rgb)*numPixels));
    
    checkCudaErrors(cudaMemcpy(inputImageDevice, inputImageHost,sizeof(Image::Rgb)*numPixels, cudaMemcpyHostToDevice));
    
    inputImageDMarker = inputImageDevice;
    // outputImageDMarker = outputImageDevice;  
}

void gaussianBlur(Image::Rgb* const inputImageDevice,
                  Image::Rgb* const outputImageDevice,
                  const size_t numRows,
                  const size_t numCols,
                  unsigned char * redBlurredDevice,
                  unsigned char * greenBlurredDevice,
                  unsigned char * blueBlurredDevice,
                  const int filterWidth){
    const dim3 blockSize(32,32,1);
    const dim3 gridSize(numCols/32,numRows/32,1);
    printf("entering the gaussianBlur\n");
    printf("numCols is %d, numRows is %d, grid: %d, %d\n", numCols, numRows, numCols/32, numRows/32);
    // separateChannels<<<gridSize,blockSize>>>(inputImageDevice,numRows,numCols,red,green,blue);
    // cudaDeviceSynchronize(); 
    // checkCudaErrors(cudaGetLastError());
 
    
    // gaussianBlur<<<gridSize, blockSize>>>(red, redBlurredDevice, numRows, numCols, filterDevice, filterWidth);
    // gaussianBlur<<<gridSize, blockSize>>>(green, greenBlurredDevice, numRows, numCols, filterDevice, filterWidth);
    // gaussianBlur<<<gridSize, blockSize>>>(blue, blueBlurredDevice, numRows, numCols, filterDevice, filterWidth);

    // cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // Image::Rgb tmpPixel;
    // combineChannels<<<gridSize, blockSize>>>(redBlurredDevice,
    //                                          greenBlurredDevice,
    //                                          blueBlurredDevice,
    //                                          outputImageDevice,
    //                                          numRows,
    //                                          numCols,tmpPixel);
    // cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}


                  
int main(int argc, const char * argv[]) {
    // insert code here...
    Image::Rgb* inputImageHost,*inputImageDevice;
    Image::Rgb* outputImageHost,*outputImageDevice;
    unsigned char* redBlurredDecive, *greenBlurredDevice, *blueBlurredDevice;
    unsigned char* red, *green, *blue;
    
    float* filterHost;
    int filterWidth;
    printf("Start of the program");
    preProcess("./lena.ppm", &inputImageHost);
    createFilterAndCopyToGPU(&filterWidth, &filterHost);
	printf("after Preprocessing\n");   
  
    allocateMemoryAndCopyToGPU(inputImageHost, inputImageDevice,outputImageDevice,redBlurredDecive,greenBlurredDevice,blueBlurredDevice);
    

    separateChannels<<<gridSize,blockSize>>>(inputImageDevice,imageInput.rows,imageInput.cols,red,green,blue);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    GpuTimer timer;
    timer.Start();
    gaussianBlur(inputImageDevice, outputImageDevice, imageInput.rows, imageInput.cols,
                redBlurredDecive, greenBlurredDevice, blueBlurredDevice, filterWidth);
    
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    timer.Stop();
    int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());
    
    if (err < 0) {
        //Couldn't print! Probably the student closed stdout - bad news
        std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
        exit(1);
    }
    
    checkCudaErrors(cudaMemcpy(outputImageHost, outputImageDevice, sizeof(Image::Rgb) * imageInput.rows*imageInput.cols, cudaMemcpyDeviceToHost));
    postProcess("result.ppm", &outputImageHost);
    
    cudaFree(inputImageDMarker);
    cudaFree(outputImageDMarker);
    checkCudaErrors(cudaFree(filterDevice));

    checkCudaErrors(cudaFree(red));
    checkCudaErrors(cudaFree(green));
    checkCudaErrors(cudaFree(blue));
    

    checkCudaErrors(cudaFree(redBlurredDecive));
    checkCudaErrors(cudaFree(greenBlurredDevice));
    checkCudaErrors(cudaFree(blueBlurredDevice));
    delete[] filterHostMarker;
    
    std::cout << "Hello, World!\n";
    return 0;
}
