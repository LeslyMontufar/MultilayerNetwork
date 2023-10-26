#include "definitions.h"
#include <random>

class Layer {
  public:
    int nNeurons;
    // int activation = 1; // 1 - Bipolar sigmoid
    std::vector<Number> w;
    std::vector<Number> y;
    std::vector<Number> dyin;
    std::vector<Number>* x;
    int nx;
    Number (*f)(Number);
    Number (*df)(Number);

    Layer(const int& nNeurons, const int& activation) : nNeurons(nNeurons) {
      this->y.resize(nNeurons);
      this->dyin.resize(nNeurons);

      switch (activation){
        case 1:
          f = f_bipolarSigmoid;
          df = df_bipolarSigmoid;
          break;

        case 2:
          f = f_binarySigmoid;
          df = df_binarySigmoid;
          break;
        
        default:
          break;
      }

    }

    Number f_bipolarSigmoid(Number& x){
      return 2/(1+std::exp(-x)) - 1;
    }

    Number df_bipolarSigmoid(Number& fx){
      return 0.5*(1+fx)*(1-fx);
    }

    Number f_binarySigmoid(Number& x){
      return 1/(1+std::exp(-x));
    }

    Number df_binarySigmoid(Number& fx){
      return fx*(1-fx);
    }

    void initWeights(std::vector<Number>& x){
      this->nx = x.size();
      this->x = &x;
      w.resize(nx*nNeurons+nNeurons); // 2*3 + 3 = 9
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<Number> dis(-0.5, 0.5);
      for (Number& value : w) {
        value = dis(gen);
      }
    }

    void calculateOut(){
      Number c;
      for(size_t j=0; j<y.size(); j++){
        c = w[nx*nNeurons+j];
        for(size_t i=0; i<nx; i++){
          c += (*x)[i] * w[i*nNeurons+j];
        }
        y[j] = f(c);
        dyin[j] = df(y[j]);
      }
    }

    friend std::ostream& operator<<(std::ostream& os, const Layer& layer) {
        size_t n = layer.w.size();
        os << "x: ";
        for(const Number x : *layer.x){
          os << x << " ";
        }
        os << "\nWeights: ";
        for (size_t i = 0; i < n-layer.nNeurons; i++) {
            os << layer.w[i] << " ";
        }
        os << "\nBias: ";
        for (size_t i = n - layer.nNeurons; i < n; i++) {
            os << layer.w[i] << " ";
        }
        return os << "\n";
    }
};

class MLP {
  private:
    std::vector<Sample> trainingSamples;
    int epochs = 10000;
    Number alpha = 0.01;
    std::vector<Layer> layers;

    Number tolerance = 1e-6;
    Number mse; // Mean Square Error

  public: 
    MLP(const std::vector<Sample>& samples) : trainingSamples(samples) {}

    // methods
    void train(){
      initLayers();

      for(size_t sampleI = 0; sampleI <samples.size(); sampleI++){
        // FeedForward
        for(size_t i=0; i<layers.size(); i++){
          layers[i].calculateOut();
        }
      }

    }

    void initLayers(){
      addLayer(Layer(samples[0].t.size(), 1));
      layers[0].initWeights(samples[0].x);
      for(size_t i=1; i<layers.size(); i++){
        layers[i].initWeights(layers[i-1].y);
      }
    }
    
    void addLayer(const Layer& layer){
      layers.push_back(layer);
    }

    void showTrainingSamples(){
      std::cout << trainingSamples;
    }

    void showTrainedNetwork(){
      for(const Layer& layer : layers) {
        std::cout << layer << "\n";
      }
    }

};

// x
// 3 neuronios
// y

int main() {
  MLP network(samples);
  // network.showTrainingSamples();
  network.addLayer(Layer(3,1));
  network.train();
  network.showTrainedNetwork();
  return 0;
}