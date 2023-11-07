#include "definitions.h"

class Sample {
public:
    // attributes
    std::vector<Number> x;
    std::vector<Number> t;
    char label;
    char labelPredicted;

    Sample(const std::vector<Number>& x, const std::vector<Number>& target, const char& label)
            : x(x), t(target), label(label) {}

    friend std::ostream& operator<<(std::ostream& os, const Sample& sample){
        os << "Entrada: ";
        for(const Number& xi : sample.x) {
            os << xi << " ";
        }
        os << "\nTarget: ";
        for(const Number& ti : sample.t) {
            os << ti << " ";
        }
        os << "\nLabel: " << sample.label << "\n\n";

        return os;
    }
};


std::vector<Sample> samplesOR = {
        Sample({-1, -1}, {-1}, '0'),
        Sample({1, -1}, {1}, '1'),
        Sample({-1, 1}, {1}, '1'),
        Sample({1, 1}, {1}, '1')
};