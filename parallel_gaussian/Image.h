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
    Image() : w(0), h(0), pixels(NULL){}
    unsigned int w, h; // image resolution
    //Rgb *pixels; // 1D array of pixels
    unsigned char* pixels;
};

#endif /* Image_h */
