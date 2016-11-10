//
//  main.cpp
//  Gaussianblur
//
//  Created by Wenwen on 11/8/16.
//  Copyright © 2016 wenwen. All rights reserved.
//




#include <iostream>
#include <fstream>
// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string>

#include "Image.h"
#include "gaussianblur.cpp"

using namespace std;
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


int main(int argc, const char * argv[]) {
    // insert code here...
    TGaussianBlur<Image::Rgb> gaussianFilter;
   Image img = readPPM("/Users/apple/Documents/forcourse/multicore computing/finalProject/ImageProcessingCUDA/GAUSSIANBLUR_CODE/Gaussianblur/Gaussianblur/lena.ppm");

    std::cout << img.pixels[1].r<<endl;
    Image imgResult;
    gaussianFilter.Filter(img.pixels, imgResult.pixels, img.h,img.w, 17);
    cout<<img.pixels[1].r<<endl;
    savePPM(img,"/Users/apple/Documents/forcourse/multicore computing/finalProject/ImageProcessingCUDA/GAUSSIANBLUR_CODE/Gaussianblur/Gaussianblur/result.ppm");
    return 0;
}
