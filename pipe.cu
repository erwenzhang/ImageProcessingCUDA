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
#include <cuda_profiler_api.h>//profile
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
        // img.pixels = (unsigned char*)malloc(sizeof(unsigned char)*3*w*h);
        cudaMallocHost((void**)&img.pixels,sizeof(unsigned char)*3*w*h);

        // new Image::Rgb[w * h]; // this is throw an exception if bad_alloc
        ifs.ignore(256, '\n'); // skip empty lines in necessary until we get to the binary data
        unsigned char pix[3];
        // read each pixel one by one and convert bytes to floats
        for (int i = 0; i < w * h; ++i) {
            ifs.read(reinterpret_cast<char *>(pix), 3);
            img.pixels[i] = pix[0];// / 255.f;
            img.pixels[w*h+i] = pix[1];// / 255.f;
            img.pixels[2*w*h+i] = pix[2];// / 255.f;
        }
        // 1d array

        ifs.close();
    }
    catch (const char *err) {
        fprintf(stderr, "%s\n", err);
        ifs.close();
    }
    return img;
}

void savePPM(unsigned char* ptr, int w, int h, const char *filename)
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
            r = static_cast<unsigned char>(std::min(255.f, ptr[i]*1.f));
            g = static_cast<unsigned char>(std::min(255.f, ptr[w*h+i]*1.f));
            b = static_cast<unsigned char>(std::min(255.f, ptr[2*w*h+i]*1.f));

            // r = static_cast<unsigned char>(std::min(255.f, ptr[i*3]*1.f));
            // g = static_cast<unsigned char>(std::min(255.f, ptr[i*3+1]*1.f));
            // b = static_cast<unsigned char>(std::min(255.f, ptr[i*3+2]*1.f));
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
    const float blurKernelSigma = 9.;
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

__global__ void gaussianBlur(unsigned char* outputChannel, const unsigned char* inputChannel, int numRows, int numCols, const float * filter, const int filterWidth)
{
    const int2 thread2DPos = make_int2(blockIdx.x*blockDim.x+threadIdx.x,blockIdx.y*blockDim.y+threadIdx.y);
    const int thread1DPos = thread2DPos.y * numCols + thread2DPos.x;
    if(thread2DPos.x >= numCols || thread2DPos.y >= numRows) return;
    float result = 0.f;
    for(int filterRow = -filterWidth/2; filterRow <= filterWidth/2; ++filterRow){
        for(int filterCol = -filterWidth/2; filterCol <= filterWidth/2; ++filterCol){
            int imageR = min(max(thread2DPos.y+filterRow,0),static_cast<int> (numRows-1));
            int imageC = min(max(thread2DPos.x+filterCol,0),static_cast<int> (numCols-1));
            unsigned char imageVal = (inputChannel[imageR*numCols + imageC]);
            float filterVal = filter[(filterRow+filterWidth/2)*filterWidth + filterCol + filterWidth/2];
            result += (static_cast<float> (imageVal))*filterVal;
        }
    }
    outputChannel[thread1DPos] = static_cast<unsigned char>(result);
}

int main(int argc, const char * argv[]) {
    // insert code here...
    Image inputImage = readPPM("lena.ppm");
    int w = inputImage.w, h = inputImage.h;

    float* filter;
    int filterWidth = 19;
    createFilter(filterWidth, &filter);
    int nStream = 3;
    cudaStream_t stream[nStream];


    // timing start
    GpuTimer timer;
    timer.Start();

    const dim3 blockSize(32,32,1);
    const dim3 gridSize(w/32,h/32,1);

    float* gpu_filter;
    cudaMalloc((void **)&gpu_filter, filterWidth*filterWidth * sizeof(float));

//while loop begin

    unsigned char *input_r, *input_g, *input_b;
    unsigned char *output_r, *output_g, *output_b;
    cudaMalloc((void **)&input_r, w*h * sizeof(unsigned char));
    cudaMalloc((void **)&input_g, w*h * sizeof(unsigned char));
    cudaMalloc((void **)&input_b, w*h * sizeof(unsigned char));
    cudaMalloc((void **)&output_r, w*h * sizeof(unsigned char));
    cudaMalloc((void **)&output_g, w*h * sizeof(unsigned char));
    cudaMalloc((void **)&output_b, w*h * sizeof(unsigned char));

    for (int i = 0; i < nStream; i++)
        cudaStreamCreate(&stream[i]);

    unsigned char* ptr = &(inputImage.pixels[0]);

    cudaMemcpy(gpu_filter, filter, filterWidth*filterWidth * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpyAsync(input_r, ptr, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(input_g, ptr+w*h, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync(input_b, ptr+2*w*h, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice, stream[2]);

    gaussianBlur<<<gridSize, blockSize, 0, stream[0]>>>(output_r, input_r, h, w, gpu_filter, filterWidth);   
    gaussianBlur<<<gridSize, blockSize, 0, stream[1]>>>(output_g, input_g, h, w, gpu_filter, filterWidth);
    gaussianBlur<<<gridSize, blockSize, 0, stream[2]>>>(output_b, input_b, h, w, gpu_filter, filterWidth);

    cudaMemcpyAsync(ptr, output_r, w*h*sizeof(unsigned char), cudaMemcpyDeviceToHost, stream[0]);
    cudaMemcpyAsync(ptr+w*h, output_g, w*h*sizeof(unsigned char), cudaMemcpyDeviceToHost, stream[1]);
    cudaMemcpyAsync(ptr+2*w*h, output_b, w*h*sizeof(unsigned char), cudaMemcpyDeviceToHost, stream[2]);


//while loop end

    cudaDeviceSynchronize(); 

    cudaFree(input_r);
    cudaFree(input_g);
    cudaFree(input_b);

    cudaFree(output_r);
    cudaFree(output_g);
    cudaFree(output_b);


    for (int i = 0; i < nStream; i++)
        cudaStreamDestroy(stream[i]);

    // timing end
    timer.Stop();
    int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());
    
    if (err < 0) {
        //Couldn't print! Probably the student closed stdout - bad news
        std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
        exit(1);
    }

    // printf("finished here\n");
    cudaFree(gpu_filter);

    savePPM(ptr, w, h,"result.ppm");
    printf("output finished\n");

    return 0;
}

