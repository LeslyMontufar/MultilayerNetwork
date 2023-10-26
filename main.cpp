#include "definitions.h"
#include <random>

class Layer {
  public:
    size_t ny; // number of out elements too
    // int activation = 1; // 1 - Bipolar sigmoid
    std::vector<Number> w;
    std::vector<Number> y;
    std::vector<Number> dyin;
    std::vector<Number>* x;
    size_t nx;
    Number (*f)(Number&);
    Number (*df)(Number&);

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

    Layer(const size_t& nNeurons, const int& activation) : ny(nNeurons) {
      this->y.resize(ny);
      this->dyin.resize(ny);

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

    void initWeights(std::vector<Number>& x){
      this->nx = x.size();
      this->x = &x;
      w.resize(nx*ny+ny); // 2*3 + 3 = 9
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
        c = w[nx*ny+j];
        for(size_t i=0; i<nx; i++){
          c += (*x)[i] * w[i*ny+j];
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
        for (size_t i = 0; i < n-layer.ny; i++) {
            os << layer.w[i] << " ";
        }
        os << "\nBias: ";
        for (size_t i = n - layer.ny; i < n; i++) {
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
    Number mse = 0; // Mean Square Error

    // aux
    Number dw;

  public: 
    MLP(const std::vector<Sample>& samples) : trainingSamples(samples) {}

    // methods
    void predict(){
      for(size_t i=0; i<layers.size(); i++){
        layers[i].calculateOut();
      }
    }

    void backPropagation(std::vector<Number> target){
      // Number dw, dE_dw, dE_dy, dz_dw;
      // dE_dy = (y - t);
      // dE_dw = dz_dw*layers.back().dyin * dE_dy;
      // dw = -alpha*dE_dw;

      // Last layer
      Number dE_dy;
      Layer layer = layers.back();
      for(size_t j=0; j<layer.ny; j++){
        dE_dy = layer.y[j]-target[j]; // errorYT
        
        for(size_t i=0; i<layer.nx; i++){
          // dE_dw = dz_dw * dy_dz*dE_dy;
          // dw = alpha*dE_dw; 
          dw = alpha* (*(layer.x))[i] *layer.dyin[j]*dE_dy;
          layer.w[i*layer.ny+j] -= dw;
        }
        layer.w[layer.nx*layer.ny+j] -= alpha*dE_dy;
        mse += dE_dy*dE_dy;
      }
      mse /= (2*layer.ny);

      // Following layers - editing here
      for(int l=layers.size()-2; l>=0; l--){
        layer = layers[l];
        // for(size_t j=0; j<layer.ny; j++){          
        //   for(size_t i=0; i<layer.nx; i++){
        //     // dE_dw = dz_dw * dy_dz*dE_dy;
        //     // dw = alpha*dE_dw; 
        //     dw = alpha* (*(layer.x))[i] *layer.dyin[j]*dE_dy;
        //     layer.w[i*layer.ny+j] -= dw;
        //   }
        //   layer.w[layer.nx*layer.ny+j] -= alpha*dE_dy;
        // }
        // dE_dy *= 
      }

    }

    void train(){
      initLayers();
      mse = 0;

      for(size_t i = 0; i <samples.size(); i++){
        // FeedForward
        layers[0].x = &samples[i].x;
        predict();

        // BackPropagation
        backPropagation(samples[i].t);
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

    void exportNetwork(){
      
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