#include "definitions.h"
#include <random>
#include <fstream>
#include <sstream>

class Layer {
  public:
    const size_t ny; // number of out elements too
    // int activation = 1; // 1 - Bipolar sigmoid
    std::vector<Number> w;
    std::vector<Number> y;
    std::vector<Number> dyin;
    const std::vector<Number>* x;
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

    void initWeights(const std::vector<Number>& x){
      this->nx = x.size();
      this->x = &x;
      w.resize(nx*ny+ny); 

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
        // os << "x: ";
        // for(const Number x : *layer.x){
        //   os << x << " ";
        // }
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
    std::vector<Sample> samples; // Training samples
    int epochs = 10000;
    Number alpha = 0.01;
    std::vector<Layer> layers;

    Number tolerance = 1e-6;
    Number mse = 0; // Mean Square Error

    Number dz_dx = 0;

  public: 
    MLP(const std::vector<Sample>& samples) : samples(samples) {}

    // methods
    void predict(){
      for(size_t i=0; i<layers.size(); i++){
        layers[i].calculateOut();
      }
    }

    void backPropagation(const std::vector<Number>& target){
      Layer* layer = &layers.back();
      std::vector<Number> dE_dy(layer->ny);
      // int s = layers.size()-1;
      Number dw, db;

      // Last layer
      for(size_t j=0; j<layer->ny; j++){
        dE_dy[j] = layer->y[j]-target[j]; // errorYT
        mse += dE_dy[j]*dE_dy[j];
      }
      mse /= (2*layer->ny);

      // BackPropagating and updating weigths
      for(int l=layers.size()-1; l>=0; l--){
        dz_dx = 0;
        layer = &layers[l];
        // std::cout << "Layer "<< l <<"/"<< s << " dE/dy: " << dE_dy << "\n";
        for(size_t j=0; j<layer->ny; j++){   
          db = alpha*dE_dy[j];
          for(size_t i=0; i<layer->nx; i++){
            dw = alpha* (*(layer->x))[i] * layer->dyin[j] * dE_dy[j];

            dz_dx += layer->w[i*layer->ny+j];

            layer->w[i*layer->ny+j] -= dw;
          }
          layer->w[layer->nx*layer->ny+j] -= db;
          // dE_dy[j] = dz_dx * layer->dyin[j] * dE_dy[j];
        }
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
      std::cout << samples;
    }

    void showTrainedNetwork(){
      for(const Layer& layer : layers) {
        std::cout << layer << "\n";
      }
    }

    void exportNetwork(){
      std::ostringstream json;
      json << "{\n";
      json << "\t\"weights\": [";

      for(const Layer& layer : layers) {
        size_t n = layer.w.size();
        json << "[";
        for (size_t i = 0; i < n; i++) {
            json << layer.w[i];
            if (i < n - 1) {
                json << ",";
            }
        }
        json << "]";
        if(&layer != &layers.back()){
          json << ",";
        }

      }
      json << "]\n";
      json << "}\n";

      std::ofstream file("../front-end/trainedNetwork.json");
      if (file.is_open()) {
        file << json.str();
        file.close();
        std::cout << "Dados salvos em trainedNetwork.json.\n";
      } else {
        std::cerr << "Erro ao abrir o arquivo para escrita.\n";
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
  network.exportNetwork();
  return 0;
}