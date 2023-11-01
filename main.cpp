#include "definitions.h"
#include "mnist.h"
#include <random>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <chrono> 

#define USE_OMP 1

class Layer {
  public:
    const size_t ny; 
    std::vector<Number> w_before;
    std::vector<Number> w;
    std::vector<Number> y;
    std::vector<Number> dyin;
    const std::vector<Number>* x;
    size_t nx;
    const act activation;
    std::vector<Number> dE_dz;
    std::vector<Number> dE_dx;

    Layer(const size_t& nNeurons, const act& activation) : ny(nNeurons), activation(activation) {
      this->y.resize(nNeurons);
      this->dyin.resize(nNeurons);
      this->dE_dz.resize(nNeurons);
    }

    void initWeights(const std::vector<Number>& vx){
      this->nx = vx.size();
      this->x = &vx;
      w.resize(nx*ny+ny); 
      dE_dx.resize(nx);

      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<Number> dis(-1./nx, 1./nx);

#if USE_OMP
#pragma omp parallel for
#endif
      for (Number& value : w) {
        value = dis(gen);
      }
    }

    void calculateOut(){
      Number c;
      for(size_t j=0; j<ny; j++){
        c = w[nx*ny+j];
#if USE_OMP
#pragma omp parallel for reduction(+ : c) 
#endif
        for(size_t i=0; i<nx; i++){
          c += (*x)[i] * w[i*ny+j];
        }
        y[j] = activation.f(c);
        dyin[j] = activation.df(y[j]);
      }
    }

    friend std::ostream& operator<<(std::ostream& os, const Layer& layer) {
        size_t n = layer.w.size();
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
    std::vector<Sample> vsamples; // Validation samples
    size_t epochs = 100000;
    Number alpha = 0.01;
    std::vector<Layer> layers;

    Number tolerance = 1e-2;
    Number mse; // Mean Square Error
    std::vector<Number> epochError;
    Number biggerdw;
    const act lastActivation;
    char (*classification)(std::vector<Number>&);
    std::vector<Number> epochWinRate;
    Number winRate;
    size_t epoch; // Epoch needed to complete the training

  public: 
    MLP(std::vector<Sample>& samples, std::vector<Sample>& vsamples, 
        const act& lastActivation, char(*classification)(std::vector<Number>&)) 
        : samples(samples), vsamples(vsamples), lastActivation(lastActivation), classification(classification) { 
      epochError.resize(epochs);
      epochWinRate.resize(epochs);
    }

    // methods
    void predict(){
#if USE_OMP
#pragma omp parallel for
#endif
      for(size_t i=0; i<layers.size(); i++){
        layers[i].calculateOut();
      }
    }

    void backPropagation(const std::vector<Number>& target){
      Layer* layer = &layers.back();
      Number errorYT, sum;

      // Last layer
      std::vector<Number>* dE_dy = &layer->y;

#pragma omp parralel for reduction(+ : mse) private(errorYT)
      for(size_t j=0; j<layer->ny; j++){
        errorYT = layer->y[j]-target[j];
        (*dE_dy)[j] = errorYT;
        mse += errorYT*errorYT;
      }
      mse /= 2;

      for(int l=layers.size()-1; l>=0; l--){
        layer = &layers[l];
        sum = 0;
#if USE_OMP
#pragma omp parallel for
#endif
        for(size_t j=0; j<layer->ny; j++){
          layer->dE_dz[j] = layer->dyin[j] * (*dE_dy)[j];
        }

        // dE_dx = dz_dx * dE_dz = w * dE_dz, w sem o b
        for(size_t i=0; i<layer->nx; i++){
          sum = 0;
#if USE_OMP
#pragma omp parallel for reduction(+ : sum)
#endif
          for(size_t j=0; j<layer->ny; j++){
            sum += layer->w[i*layer->ny+j]*layer->dE_dz[j];
          }
          layer->dE_dx[i] = sum;
        }

        dE_dy = &layer->dE_dx;

        for(size_t j=0; j<layer->ny; j++){

#pragma omp parallel for
          for(size_t i=0; i<layer->nx; i++){
            layer->w[i*layer->ny+j] -= alpha * (*(layer->x))[i] * layer->dE_dz[j];
          }
          layer->w[layer->nx*layer->ny+j] -= alpha * layer->dE_dz[j];
        }
      }     
    }

    void stopCondition(){
#pragma omp parallel for reduction(max:biggerdw)
      for(Layer& layer : layers){
        for(size_t i=0; i<layer.w.size(); i++){
          layer.w_before[i] = std::abs(layer.w[i]-layer.w_before[i]);
          biggerdw = std::max(layer.w_before[i], biggerdw);
        }
      }
    }

    void validation(std::vector<Sample>& samples){
#if USE_OMP
#pragma omp parallel for
#endif
      for(Sample& sample : samples){
        layers[0].x = &sample.x;
        predict();
        sample.labelPredicted = classification(layers.back().y);
        if(sample.labelPredicted == sample.label){
          winRate+=1;
        }
      }
      winRate = winRate/samples.size() *100;
    }

    void progressBar(const int& percent, const int& win){
        std::cout << "\r[";
        for (int i = 0; i < win; i++) {
          std::cout << char(254); 
        }
        for(int i=win; i<100; i++) {
          std::cout << " ";
        }
        std::cout << "] " << percent << "% "<< win << "%";
        std::cout.flush();
    }

    void train(){
      initLayers();
      std::cout << "\n";
      for(epoch = 0; epoch < epochs; epoch++){ 
#if USE_OMP
#pragma omp parallel for
#endif  
        for(Layer& layer : layers){
          layer.w_before = layer.w;
        }
        biggerdw = 0;
        mse = 0;
        winRate = 0;
        for(size_t i = 0; i <samples.size(); i++){
          // FeedForward
          layers[0].x = &samples[i].x;
          predict();

          // BackPropagation
          backPropagation(samples[i].t);
          epochError[epoch] += mse;
          
        }
        epochError[epoch] /= samples.size();
        stopCondition();
        validation(samples);
        epochWinRate[epoch] = winRate;
        progressBar((epoch/(Number)epochs)*100, winRate);
        if((biggerdw <= tolerance) && (winRate>90)){
          break;
        }
        
      }
      std::cout << "\n\nTreinamento concluido apos " << epoch << " epocas.\n"; //sem +1
      std::cout << "WinRate: " << winRate << "%\tMSE: " << epochError[epoch-1] << "\n\n";
      
    }

    void initLayers(){
      addLayer(Layer(samples[0].t.size(), lastActivation));
      layers[0].initWeights(samples[0].x);
      for(size_t i=1; i<layers.size(); i++){
        layers[i].initWeights(layers[i-1].y);
      }
    }
    
    void addLayer(const Layer& layer){
      layers.push_back(layer);
    }

    void showTrainingSamples(){
      for(Sample& sample : samples){
        std::cout << sample;
      }
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

      std::ofstream file("front-end/trainedNetwork.json");
      if (file.is_open()) {
        file << json.str();
        file.close();
        std::cout << "Dados salvos em trainedNetwork.json.\n";
      } else {
        std::cerr << "Erro ao abrir o arquivo para escrita.\n";
      }

    }

    void showResults(std::vector<Sample>& s){
      validation(s);
      std::cout << "Label informed:  ";
#pragma omp parallel for
      for(const Sample& sample : s){
        std::cout << sample.label << " ";
      }
      std::cout << "\nLabel predicted: ";
#pragma omp parallel for
      for(const Sample& sample : s){
        std::cout << sample.labelPredicted << " ";
      }
      std::cout << "\n";
    }
};

int main() {
  
  const char *imageFile = "input/train-images.idx3-ubyte";
  const char *labelFile = "input/train-labels.idx1-ubyte";
  const char *testImageFile = "input/t10k-images.idx3-ubyte";
  const char *testLabelFile = "input/t10k-labels.idx1-ubyte";
  
  std::vector<Sample> samples, vsamples;
  std::cout << "Erro: " << loadData(imageFile, labelFile, samples, -1, 100) << "\n";
  // std::cout << "Erro: " << loadData(testImageFile, testLabelFile, vsamples, 0, 5) << "\n";
  
  MLP network(samples, vsamples, linear,
              [](std::vector<Number>& y) -> char {
                for(char i=0; i<(char)y.size(); i++){
                  if(y[i]>=0){
                    return i + '0';
                  } 
                } 
                return 0;
              });

  std::cout << "Quatidade de amostras de treinamento: " << samples.size() << "\n";
  // std::cout << "Quatidade de amostras de teste: " << vsamples.size() << "\n";
  
  network.addLayer(Layer(100,bipolarSigmoid));
  
  auto start_time = std::chrono::high_resolution_clock::now(); 
  network.train();
  auto end_time = std::chrono::high_resolution_clock::now(); 
  
  std::chrono::duration<double> duration = end_time - start_time; 
  std::cout << "Training finished after " << duration.count() <<" seconds\n\n";
  
  network.showResults(samples);
  std::cout << "\n\n";
  // network.exportNetwork();
  return 0;
}