#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <iostream>
#include <vector>

typedef double Number;

class Sample {       
  public:
  // attributes
    std::vector<Number> x;
    std::vector<Number> t;
    int label;

    Sample(const std::vector<Number>& x, const std::vector<Number>& target, const int& label)
            : x(x), t(target), label(label) {}
  
  // friend permite que a função os membros privados da classe Sample
  // mas quero que a MLP tbm tenha acesso entao declarei publico
//   friend std::ostream& operator<<(std::ostream& os, const Sample& sample);
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

std::vector<Sample> samples = {
    Sample({-1, -1}, {-1}, 0),
    Sample({1, -1}, {1}, 1),
    Sample({-1, 1}, {1}, 1),
    Sample({1, 1}, {1}, 1)
};

#endif