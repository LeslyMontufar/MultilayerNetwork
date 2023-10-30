#ifndef MNIST_CPP_H
#define MNIST_CPP_H

#include <iostream>
#include "definitions.h"
#include <cstdint>
#include <vector>
#include <string>

using namespace std;

#define MNIST_MAGIC_NUMBER_LITTLE  0x03080000
#define MNIST_MAGIC_NUMBER_BIGENDIAN  0x00000803
#define MNIST_LABEL_MAGIC_NUMBER_LITTLE 0x01080000
#define MNIST_LABEL_MAGIC_NUMBER_BIGENDIAN 0x00000801

uint32_t read_int_big2little(FILE *f) {
    uint32_t num;
    fread(&num, 4, 1, f);
    return ((num >> 24) & 0xFF) |
           ((num >> 8) & 0xFF00) |
           ((num << 8) & 0xFF0000) |
           ((num << 24) & 0xFF000000);
}

uint32_t read_int_little(FILE *f) {
    uint32_t num;
    fread(&num, 4, 1, f);
    return num;
}

int loadData(const char* imageFile, const char* labelFile, vector<Sample>& samples, const Number zero_representation = 0, const uint32_t nImages = 0) {
    FILE *fimage, *flabel;
    uint32_t (*read_int32)(FILE *);
    uint32_t nImageMNIST;
    int mnist_w;
    int mnist_h;
    int mnist_im_size;
    uint32_t nLabel;

    fimage = fopen(imageFile, "rb");
    if (!fimage) return -1;

    flabel = fopen(labelFile, "rb");
    if (!flabel) {
        fclose(fimage);
        return -1;
    }

    read_int32 = read_int_little;
    nImageMNIST = read_int_little(fimage); // get magic number
    
    if (nImageMNIST == MNIST_MAGIC_NUMBER_LITTLE) {
        read_int32 = read_int_big2little;
    } else if (nImageMNIST != MNIST_MAGIC_NUMBER_BIGENDIAN) {
        return -2;
    }

    nLabel = read_int_little(flabel);
    if (nLabel == MNIST_LABEL_MAGIC_NUMBER_LITTLE) {
        read_int32 = read_int_big2little;
    } else if (nLabel != MNIST_LABEL_MAGIC_NUMBER_BIGENDIAN) {
        return -2;
    }

    nImageMNIST = read_int32(fimage);
    nLabel = read_int32(flabel);
    if (nImageMNIST != nLabel) {
        return -3;
    }
    
    if(nImages){
        nImageMNIST = nImages;
        nLabel = nImages;
    }

    mnist_w = read_int32(fimage);
    mnist_h = read_int32(fimage);
    mnist_im_size = mnist_h * mnist_w;
    
    char label;
    unsigned char* im;
    im = new unsigned char[mnist_h * mnist_w];
    for (uint32_t n = 0; n < nImageMNIST; ++n) {
        fread(&label, 1, 1, flabel);

        samples.push_back(Sample(std::vector<Number>(mnist_im_size), std::vector<Number>(10), label));
        Sample& sample = samples.back();
        sample.label = '0'+label;
        fread(im, 1, mnist_im_size, fimage);

        for (int i = 0; i < mnist_im_size; ++i) {
            sample.x[i] = im[i] ? 1 : -1;
        }
        for (unsigned char i = 0; i < 10; ++i) {
            sample.t[i] = (label == i) ? 1 : zero_representation;
        }
        
    }

    delete[] im;
    fclose(flabel);
    fclose(fimage);
    return 0;
}

#endif //MNIST_CPP_H
