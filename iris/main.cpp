#include "iris.h"
#include "main.h"

int main() {
    MLP network(samplesOR, samplesOR, linear,
              [](std::vector<Number>& y) -> char {
                return (y[0]>=0)? '1' : '0';
              });

    network.addLayer(Layer(2,bipolarSigmoid));


    auto start_time = std::chrono::high_resolution_clock::now();
    network.train();
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Training finished after " << duration.count() <<" seconds\n";

    network.updateMe(0);

    return 0;
}