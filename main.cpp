#include "definitions.h"
#include <random>

class Layer {
  public:
    int activation = 1; // 1 - Bipolar sigmoid
    std::vector<Number> w;

    Layer(const int& nNeurons, const int& activationF) : activation(activationF) {
        // Inicialize o gerador de números aleatórios
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<Number> dis(-0.5, 0.5);

        // Redimensione o vetor w e preencha com números aleatórios
        w.resize(nNeurons + 1);  // +1 para o viés (bias)
        for (int i = 0; i < (int) w.size(); i++) {
            w[i] = dis(gen);
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Layer& layer) {
        os << "Activation: " << layer.activation << "\nWeights: ";
        for (const Number& weight : layer.w) {
            os << weight << " ";
        }
        return os;
    }
};

class MLP {
  private:
    std::vector<Sample> trainingSamples;
    int epochs = 10000;
    Number alpha = 0.01;
    std::vector<Layer> layers;

  public: 
    MLP(const std::vector<Sample>& samples) : 
        trainingSamples(samples) {}

    // methods
    void showTrainingSamples(){
      std::cout << trainingSamples;
    }

    void train(){

    }
    void showTrainedNetwork(){
      for(const Layer& layer : layers) {
        std::cout << layer << "\n";
      }
        
    }

    void addLayer(const Layer& layer){
      layers.push_back(layer);
    }

};

int main() {
  MLP network(samples);
  // network.showTrainingSamples();
  network.addLayer(Layer(3,1));
  // network.train();
  network.showTrainedNetwork();
  return 0;
}