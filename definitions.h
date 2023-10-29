#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <iostream>
#include <vector>
#include <cmath>

typedef double Number;

class Sample {       
  public:
  // attributes
    const std::vector<Number> x;
    const std::vector<Number> t;
    const std::string label;

    Sample(const std::vector<Number>& x, const std::vector<Number>& target, const std::string& label)
            : x(x), t(target), label(label) {}
  
    friend std::ostream& operator<<(std::ostream& os, const std::vector<Sample>& samples){
        for(const Sample& sample : samples){
            os << "Entrada: ";
            for(const Number& xi : sample.x) {
                os << xi << " ";
            }
            os << "\nTarget: ";
            for(const Number& ti : sample.t) {
                os << ti << " ";
            }
            os << "\nLabel: ";
            os << sample.label;
            os << "\n\n";
        }
        return os;
    }
};

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

std::ostream& operator<<(std::ostream& os, const std::vector<Number>& v){
    for(const Number& value : v){
        os << value << " ";
    }
    return os;
}

std::vector<Sample> samples = {
    Sample({-1, -1}, {-1}, "0"),
    Sample({1, -1}, {1}, "1"),
    Sample({-1, 1}, {1}, "1"),
    Sample({1, 1}, {1}, "1")
};

#endif