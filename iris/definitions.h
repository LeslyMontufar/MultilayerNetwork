#ifndef MULTILAYERNETWORK_DEFINITIONS_H
#define MULTILAYERNETWORK_DEFINITIONS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstdio>

#define USE_OMP 1

typedef double Number;

static Number f_bipolarSigmoid(Number& x){
    return 2/(1+std::exp(-x)) - 1;
}

static Number df_bipolarSigmoid(Number& fx){
    return 0.5*(1+fx)*(1-fx);
}

static Number f_binarySigmoid(Number& x){
    return 1/(1+std::exp(-x));
}

static Number df_binarySigmoid(Number& fx){
    return fx*(1-fx);
}

struct act {
    Number (*f)(Number&);
    Number (*df)(Number&);
};

const act bipolarSigmoid = {
        f_bipolarSigmoid,
        df_bipolarSigmoid
};

const act binarySigmoid = {
        f_binarySigmoid,
        df_binarySigmoid
};

const act linear = {
        [](Number& x){return x;},
        [](Number& ){return (Number)1;}
};

const act bipolarStep = {
        [](Number& x){return (Number) ((x>=0)? 1:-1);},
        [](Number& ){return (Number) 0;}
};

const act binaryStep = {
        [](Number& x){return (Number) ((x>=0)? 1:0);},
        [](Number& ){return (Number) 0;}
};

std::ostream& operator<<(std::ostream& os, const std::vector<Number>& v){
    for(const Number& value : v){
        os << value << " ";
    }
    return os;
}

#endif //MULTILAYERNETWORK_DEFINITIONS_H
