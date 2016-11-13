//
//  main.cpp
//  GaussianBlurWithCuda
//
//  Created by Wenwen on 11/10/16.
//  Copyright © 2016 wenwen. All rights reserved.
//

#include <iostream>
#include <string>
#include <stdio.h>

#include "Image.h"
#include "utils.h"
#include "timer.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gaussianBlur(const unsigned char* inputChannel, unsigned char* const outputChannel, int numRows, int numCols, const float* const filter, const int filterWidth){
    const int2 thread2DPos = make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    const int thread1DPos = thread2DPos.y * numCols + thread2DPos.x;
    
    if(thread2DPos.y >= numCols ||thread2DPos.x >= numRows) return;
    
    float result = 0.f;
    for(int filterRow = -filterWidth/2; filterRow <= filterWidth/2; ++filterRow){
        for(int filterCol = -filterWidth/2; filterCol <= filterWidth/2; ++filterCol){
            int imageR = min(max(thread2DPos.x+filterRow,0),static_cast<int> (numRows-1));
            int imageC = min(max(thread2DPos.y+filterCol,0),static_cast<int> (numCols-1));
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
                                                              
    if(thread2DPos.y >= numCols ||thread2DPos.x >= numRows) return;
    redChannel[thread1DPos] = inputImageDevice[thread1DPos].r;
    greenChannel[thread1DPos] = inputImageDevice[thread1DPos].g;
    blueChannel[thread1DPos] = inputImageDevice[thread1DPos].b;
}
                                                              
__global__ void combineChannels(const unsigned char* const redChannel,const unsigned char* const greenChannel, const unsigned char* const blueChannel, Image::Rgb* const outputImageDevice, int numRows,int numCols){
    const int2 thread2DPos = make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    const int thread1DPos = thread2DPos.y * numCols + thread2DPos.x;
                                                              
    if(thread2DPos.y >= numCols ||thread2DPos.x >= numRows) return;
                                                              
    unsigned char red = redChannel[thread1DPos];
    unsigned char green = greenChannel[thread1DPos];
    unsigned char blue = blueChannel[thread1DPos];
    Image::Rgb pixel;
    pixel.r = red; 
    pixel.g = green;
    pixel.b = blue;
    outputImageDevice[thread1DPos] = pixel;
}

unsigned char *red,*green,*blue;
float *filterDevice;

void allocateMemoryAndCopyToGPU(const size_t numRows, const size_t numCols, const float* const filterHost, const size_t filterWidth){
    checkCudaErrors(cudaMalloc(&red,sizeof(unsigned char)*numRows*numCols));
    checkCudaErrors(cudaMalloc(&green,sizeof(unsigned char)*numRows*numCols));
    checkCudaErrors(cudaMalloc(&blue,sizeof(unsigned char)*numRows*numCols));
    
    checkCudaErrors(cudaMalloc(&filterDevice,sizeof(float)*filterWidth*filterWidth));
    
    checkCudaErrors(cudaMemcpy(filterDevice,filterHost,sizeof(float)*filterWidth*filterWidth,cudaMemcpyHostToDevice));
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
    const dim3 gridSize(numCols/32+1,numRows/32+1,1);
    
    separateChannels<<<gridSize,blockSize>>>(inputImageDevice,numRows,numCols,red,green,blue
                                             );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    gaussianBlur<<<gridSize, blockSize>>>(red, redBlurredDevice, numRows, numCols, filterDevice, filterWidth);
    gaussianBlur<<<gridSize, blockSize>>>(green, greenBlurredDevice, numRows, numCols, filterDevice, filterWidth);
    gaussianBlur<<<gridSize, blockSize>>>(blue, blueBlurredDevice, numRows, numCols, filterDevice, filterWidth);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    combineChannels<<<gridSize, blockSize>>>(redBlurredDevice,
                                            greenBlurredDevice,
                                            blueBlurredDevice,
                                            outputImageDevice,
                                            numRows,
                                            numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
   
}


Image imageInput;
Image imageOutput;
Image::Rgb* inputImageDMarker;
Image::Rgb* outputimageDMarker;
float* filterHostMarker;
void loadImagePPM(Image imageInput, Image imageOutput, const char* filename){
    FILE * fptr = fopen(filename, "r"); 
    if (!fptr) {
        printf("!! Error in opening data file \n"); 
        exit(1); 
    }
    std::string type;
    int w, h, x;
    fscanf(fptr, "%s",type);
    fscanf(fptr,"%d",w);
    fscanf(fptr,"%d",h);
    fscanf(fptr,"%d",x);
    char pix[3];
    imageInput.rows = h;
    imageInput.cols = w;
    imageOutput.pixels = new Image::Rgb[w*h];
    imageInput.pixels = new Image::Rgb[w*h];
    for(int i = 0; i < w*h; i++){
      fgets(pix, 3, (FILE*)fptr);
      imageInput.pixels[i].r = static_cast<unsigned char>(pix[0]);
      imageInput.pixels[i].g = static_cast<unsigned char>(pix[1]);
      imageInput.pixels[i].b = static_cast<unsigned char>(pix[2]);
    }
  
    
    fclose(fptr); 
}

void saveImagePPM(const char* filename, Image::Rgb* outputImageHost){
    int w = imageInput.cols;
    int h = imageInput.rows;
    FILE * fptr = fopen(filename, "w"); 
    fputs("P6\n",fptr);
    fputs(w+"\n",fptr);
    fputs(h+"\n",fptr);
    fputs("255\n",fptr);

    int r,g,b;
    for(int i = 0; i < w*h; i++){
      r = static_cast<int>(outputImageHost[i].r);
      g = static_cast<int>(outputImageHost[i].g);
      b = static_cast<int>(outputImageHost[i].b);
      fputc(r,fptr);
      fputc(g,fptr);
      fputc(b,fptr);
    }
      
    fclose(fptr); 
}

void preProcess(const char *filename, Image::Rgb ** inputImageHost, Image::Rgb ** outputImageHost, Image::Rgb ** inputImageDevice, Image::Rgb **outputImageDevice, unsigned char ** redBlurredDevice, unsigned char ** greenBlurredDevice, unsigned char **blueBlurredDevice){
    
    checkCudaErrors(cudaFree(0));
   
    loadImagePPM(imageInput, imageOutput, filename);
 
    
    *inputImageHost = (Image::Rgb *)imageInput.pixels;
    *outputImageHost = (Image::Rgb *)imageOutput.pixels;
    
    const size_t numPixels = imageInput.rows*imageInput.cols;
    
    checkCudaErrors(cudaMalloc(inputImageDevice,sizeof(Image::Rgb)*numPixels));
    checkCudaErrors(cudaMalloc(outputImageDevice,sizeof(Image::Rgb)*numPixels));
    checkCudaErrors(cudaMemset(*outputImageDevice,0,sizeof(Image::Rgb)*numPixels));
    
    checkCudaErrors(cudaMemcpy(*inputImageDevice, *inputImageHost,sizeof(Image::Rgb)*numPixels, cudaMemcpyHostToDevice));
    
    inputImageDMarker = *inputImageDevice;
    outputimageDMarker = *outputImageDevice;
    
    checkCudaErrors(cudaMalloc(redBlurredDevice,    sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc(greenBlurredDevice,  sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc(blueBlurredDevice,   sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(*redBlurredDevice,   0, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(*greenBlurredDevice, 0, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(*blueBlurredDevice,  0, sizeof(unsigned char) * numPixels));
    
}



void createFilter(int* filterWidth, float** filterHost){
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
    
}

void postProcess(const char* output_file, Image::Rgb* outputImageHost) {
    saveImagePPM(output_file, outputImageHost);
}

void cleanUp(void){
    cudaFree(inputImageDMarker);
    cudaFree(outputimageDMarker);
    
    checkCudaErrors(cudaFree(red));
    checkCudaErrors(cudaFree(green));
    checkCudaErrors(cudaFree(blue));
    checkCudaErrors(cudaFree(filterDevice));
    
    delete[] filterHostMarker;
}
                  
int main(int argc, const char * argv[]) {
    // insert code here...
    Image::Rgb* inputImageHost,*inputImageDevice;
    Image::Rgb* outputImageHost,*outputImageDevice;
    unsigned char* redBlurredDecive, *greenBlurredDevice, *blueBlurredDevice;
    
    float* filterHost;
    int filterWidth;
    
    preProcess("./lena.ppm", &inputImageHost, &outputImageHost, &inputImageDevice, &outputImageDevice, &redBlurredDecive, &greenBlurredDevice, &blueBlurredDevice);
    
    createFilter(&filterWidth, &filterHost);
    
    allocateMemoryAndCopyToGPU(imageInput.rows, imageInput.cols, filterHost, filterWidth);
    
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
    postProcess("result", outputImageHost);
    
    checkCudaErrors(cudaFree(redBlurredDecive));
    checkCudaErrors(cudaFree(greenBlurredDevice));
    checkCudaErrors(cudaFree(blueBlurredDevice));
    cleanUp();
    
    std::cout << "Hello, World!\n";
    return 0;
}
