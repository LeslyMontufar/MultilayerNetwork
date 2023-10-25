#ifndef SAMPLES_H
#define SAMPLES_H

#include <vector>
typedef double Number;

class Sample {       
  public:
  // attributes
    std::vector<Number> x;
    std::vector<Number> t;
    int label;

    Sample(const std::vector<Number>& x, const std::vector<Number>& target)
            : x(x), t(target) {}
  
  // methods
};

std::vector<Sample> samples = {
    Sample({0.1, 0.2, 0.3}, {0.4, 0.5}),
    Sample({0.2, 0.3, 0.4}, {0.5, 0.6}),
    // Adicione mais amostras conforme necessário
};

#endif