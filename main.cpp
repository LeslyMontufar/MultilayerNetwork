#include "definitions.h"
#include "samples.h"

class MLP {
  private:
    std::vector<Sample> trainingSample;

  public: 
    MLP(const std::vector<Sample>& samples) : trainingSample(samples) {}

    // methods
    void showInfo(){
      std::cout << trainingSample;
    }

};

int main() {
  MLP network(samples);
  network.showInfo();

  return 0;
}