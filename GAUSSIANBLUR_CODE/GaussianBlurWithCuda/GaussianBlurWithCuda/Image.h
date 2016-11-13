//
//  Image.h
//  Gaussianblur
//
//  Created by Wenwen on 11/9/16.
//  Copyright © 2016 wenwen. All rights reserved.
//

#ifndef Image_h
#define Image_h

class Image
{
public:
    // Rgb structure, i.e. a pixel
    struct Rgb
    {
        // Rgb() : r(0), g(0), b(0)  {}
        // Rgb(float c) : r(c), g(c), b(c) {}
        // Rgb(float _r, float _g, float _b) : r(_r), g(_g), b(_b) {}
        // bool operator != (const Rgb &c) const { return c.r != r || c.g != g || c.b != b; }
        // Rgb& operator *= (const Rgb &rgb) { r *= rgb.r, g *= rgb.g, b *= rgb.b; return *this; }
        // Rgb& operator += (const Rgb &rgb) { r += rgb.r, g += rgb.g, b += rgb.b; return *this; }
        // friend float& operator += (float &f, const Rgb rgb)
        // { f += (rgb.r + rgb.g + rgb.b) / 3.f; return f; }
        unsigned char r, g, b;
    };
    
    Image() : w(0), h(0), pixels(NULL)
    { /* empty image */ }
//    Image(const unsigned int &_w, const unsigned int &_h, const Rgb &c = kBlack) : w(_w), h(_h), pixels(NULL)
//    {
//        pixels = new Rgb[w * h];
//        for (int i = 0; i < w * h; ++i) pixels[i] = c;
//    }
    const Rgb& operator [] (const unsigned int &i) const { return pixels[i]; }
    Rgb& operator [] (const unsigned int &i) { return pixels[i]; }
//    ~Image() {
//        //        if (pixels != nullptr){
//        //            delete[] pixels;
//        //            pixels = nullptr;
//        //        }
//    }
    unsigned int w, h; // image resolution
    unsigned int cols,rows;
    Rgb *pixels; // 1D array of pixels
    static const Rgb kBlack, kWhite, kRed, kGreen, kBlue; // preset colors
};

#endif /* Image_h */
