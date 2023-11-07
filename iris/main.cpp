#include "MLP.h"
#include "iris.h"

int main() {
  MLP network(samples, samples, linear,
            [](std::vector<Number>& y) -> char {
              return '0' + std::round(y[0]);
            });

  network.addLayer(Layer(10,bipolarSigmoid));
  network.addLayer(Layer(50,bipolarSigmoid));
  network.addLayer(Layer(10,bipolarSigmoid));


  auto start_time = std::chrono::high_resolution_clock::now();
  network.train();
  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end_time - start_time;
  std::cout << "Training finished after " << duration.count() <<" seconds\n";

  network.saveNetwork();

  return 0;
}