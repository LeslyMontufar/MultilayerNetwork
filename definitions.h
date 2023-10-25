#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <iostream>
#include <vector>
#include "samples.h"

typedef double Number;

std::ostream& operator<<(std::ostream& os, const std::vector<Sample>& samples){
    for(const Sample& sample : samples){
        os << "Entrada: ";
        for(const Number& xi : sample.x) {
            os << xi << " ";
        }
        os << "\nTarget: ";
        for(const Number& ti : sample.t) {
            os << ti << " ";
        }
        os << "\n\n";
    }
}

#endif