#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstring>
#include <algorithm> 
#include "Image.h"
#include "math.h"
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

void savePPM(const Image &img, const char *filename)
{

    if (img.w == 0 || img.h == 0) { fprintf(stderr, "Can't save an empty image\n"); return; }
    std::ofstream ofs;
    try {
        ofs.open(filename, std::ios::binary); // need to spec. binary mode for Windows users
        if (ofs.fail()) throw("Can't open output file");
        ofs << "P6\n" << img.w << " " << img.h << "\n255\n";
        unsigned char r, g, b;
        // loop over each pixel in the image, clamp and convert to byte format
        for (int i = 0; i < img.w * img.h; ++i) {
            r = static_cast<unsigned char>(std::min(1.f, img.pixels[i].r) * 255);
            g = static_cast<unsigned char>(std::min(1.f, img.pixels[i].g) * 255);
            b = static_cast<unsigned char>(std::min(1.f, img.pixels[i].b) * 255);
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

int main(int argc, const char * argv[]) {
    // insert code here...
    Image inputImage = readPPM("lena.ppm");

    float* filter;

    int filterWidth = 9;

    createFilter(filterWidth, &filter);

    Image outputImage;
    outputImage.w = inputImage.w; outputImage.h = inputImage.h;
    outputImage.pixels = new Image::Rgb[inputImage.w * inputImage.h];  

    gaussianBlur(&inputImage, &outputImage, inputImage.h, inputImage.w, filter, filterWidth);

    savePPM(outputImage,"result.ppm");

    return 0;
}
