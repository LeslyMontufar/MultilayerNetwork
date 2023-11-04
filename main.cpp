#include "definitions.h"
#include "mnist.h"
#include <fstream>
#include <sstream>
#include <chrono> 
#include <cstdio>
#include "Layer.h"
#include "w.h"

class MLP {
  private:
    std::vector<Sample> samples; // Training samples
    std::vector<Sample> vsamples; // Validation samples
    size_t epochs = 20;
    Number alpha = 0.01;
    std::vector<Layer> layers;

    Number mse; // Mean Square Error
    std::vector<Number> epochError;
    Number biggerdw;
    const act lastActivation;
    char (*classification)(std::vector<Number>&);
    std::vector<Number> epochWinRate;
    Number winRate;
    size_t epoch; // Epoch needed to complete the training
    std::vector<int> confusionTable;

  public: 
    MLP(std::vector<Sample>& samples, std::vector<Sample>& vsamples, 
        const act& lastActivation, char(*classification)(std::vector<Number>&)) 
        : samples(samples), vsamples(vsamples), lastActivation(lastActivation), classification(classification) { 
      epochError.resize(epochs);
      epochWinRate.resize(epochs);
      confusionTable.resize(100);
    }

    // methods
    void predict(){
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

    void validation(std::vector<Sample>& samples){
      winRate = 0;
#pragma omp parallel for
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

    void progressBar(const int& epochPercent, const int& samplePercent){
        std::cout << "\r[";
#pragma omp parallel for
        for (int i = 0; i < samplePercent; i++) {
          std::cout << char(254); 
        }
#pragma omp parallel for
        for(int i=samplePercent; i<100; i++) {
          std::cout << " ";
        }
        std::cout << "] " << epochPercent << "% " << samplePercent << "% ";
        std::cout.flush();
    }

    void progressBarSample(const int& samplePercent, const int& win){
        std::cout << "\r[";
#pragma omp parallel for
        for (int i = 0; i < samplePercent; i++) {
          std::cout << char(254); 
        }
#pragma omp parallel for
        for(int i=samplePercent; i<100; i++) {
          std::cout << " ";
        }
        std::cout << "] " << samplePercent << "% " << win << "% ";
        std::cout.flush();
    }

    void train(){
      size_t nsamples = samples.size();
      initLayers();
      std::cout << "\n";
      for(epoch = 0; epoch < epochs; epoch++){ 
        mse = 0;
        // winRate = 0;
        for(size_t i = 0; i < nsamples; i++){
          progressBar(((epoch+1)/(Number)epochs)*100, (i+1)/(Number) nsamples *100);
          // FeedForward
          layers[0].x = &samples[i].x;
          predict();

          // BackPropagation
          backPropagation(samples[i].t);
          epochError[epoch] += mse;
          
        }
        epochError[epoch] /= nsamples;
        validation(vsamples); // Validation with validation samples
        epochWinRate[epoch] = winRate;

        std::cout << (int) winRate << "%";
        if(winRate>=100){ 
          break;
        }
        else if((int)winRate > (int)epochWinRate[epoch-1] && (int)winRate > 90){
          updateMe(winRate);
        }
      }
      std::cout << "\n\nTreinamento concluido apos " << epoch << " epocas.\n"; //sem +1
      std::cout << "WinRate: " << winRate << "%\tMSE: " << epochError[epoch-1] << "\n\n";
      
    }

    void detailedResult(){
      winRate = 0;
      for(Sample& sample :samples){
        layers[0].x = &sample.x;
        predict();
        sample.labelPredicted = classification(layers.back().y);
        if(sample.labelPredicted == sample.label){
          winRate+=1;
        }
        std::cout << sample.label << "\ttarget: " << sample.t << "\n" << sample.labelPredicted << "\ty\t: " << layers.back().y << "\n\n";
      }
      std::cout << "WinRate final: " << winRate/samples.size()*100 << "%\n\n";;
        
    }

    void updateMe(Number tag){
        std::cout << "\n\nValidation: " << tag << "  \n\n";
        showResults(samples);
        std::cout << "\n";
        showResults(vsamples);
        std::cout << "\n";
        if(!tag) tag = winRate;
        exportNetwork(tag);
        saveNetwork();
        std::cout << "\n\n";
    }

    void initLayers(){
      addLayer(Layer(samples[0].t.size(), lastActivation));
      layers[0].initWeights(samples[0].x);
#pragma omp parallel for
      for(size_t i=1; i<layers.size(); i++){
        layers[i].initWeights(layers[i-1].y);
      }
    }
    
    void addLayer(const Layer& layer){
      layers.push_back(layer);
    }

    void exportNetwork(const Number& tag){
      char filename[50];
      snprintf(filename, sizeof(filename), "front-end/w%.0f.js", tag);

      std::ostringstream content;
      content << "const w = [";

      for(const Layer& layer : layers) {
        size_t n = layer.w.size();
        content << "[";
        for (size_t i = 0; i < n; i++) {
            content << layer.w[i];
            if (i < n - 1) {
              content << ",";
            }
        }
        content << "]";
        if(&layer != &layers.back()){
          content << ",";
        }

      }
      content << "]\n";

      std::ofstream file(filename);
      if (file.is_open()) {
        file << content.str();
        file.close();
        std::cout << "Dados salvos em " << filename << ".\n";
      } else {
        std::cerr << "Erro ao abrir o arquivo para escrita.\n";
      }

    }

    void showResults(std::vector<Sample>& s){
#pragma omp parallel for
      for(int& c : confusionTable){
        c = 0;
      }
      // validation com progress bar
#pragma omp parallel for
      winRate = 0;
      Sample* sample;
      size_t ssize = s.size();
      for(size_t i=0; i<ssize; i++){
        sample = &s[i];
        layers[0].x = &sample->x;
        predict();
        sample->labelPredicted = classification(layers.back().y);
        if(sample->labelPredicted == sample->label){
          winRate+=1;
        }
        confusionTable[(sample->label-'0')*10 + sample->labelPredicted - '0'] += 1;
        progressBarSample((i+1)/(Number)ssize*100, winRate/ssize * 100);
      }
      std::cout << "\n";

      for(int i=0; i<10; i++){
        std::cout << i << ": ";
#pragma omp parallel for
        for(int j=0; j<10; j++){
          std::cout << confusionTable[i*10+j] << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }

    void saveNetwork(){
      std::ostringstream content;
      content << "#ifndef W_H\n";
      content << "#define W_H\n";
      content << "#include \"definitions.h\"\n";
      content << "std::vector<std::vector<Number>> w = {\n";
      
      for(size_t i=0; i<layers.size(); i++){
        content << "{"; 
        for(int iw=0; iw<layers[i].w.size()-1; iw++){
          content << layers[i].w[iw] << ", ";
        }
        content << layers[i].w.back();
        content << "}";
        if(i<layers.size()-1){
          content << ",\n";
        } else{
          content << "\n";
        }
      }
      content << "};\n";
      content << "#endif\n";

      std::ofstream file("w.h");
      if (file.is_open()) {
        file << content.str();
        file.close();
        std::cout << "Dados salvos em w.h.\n";
      } else {
        std::cerr << "Erro ao abrir o arquivo de w para escrita.\n";
      }
      
    }

};

int main() {

#if 1
  const char *imageFile = "input/train-images.idx3-ubyte";
  const char *labelFile = "input/train-labels.idx1-ubyte";
  const char *testImageFile = "input/t10k-images.idx3-ubyte";
  const char *testLabelFile = "input/t10k-labels.idx1-ubyte";

  std::vector<Sample> samples, vsamples;
  std::cout << "Erro: " << loadData(imageFile, labelFile, samples, -1, 0) << "\n";
  std::cout << "Erro: " << loadData(testImageFile, testLabelFile, vsamples, -1, 0) << "\n";
  
  MLP network(samples, vsamples, linear,
              [](std::vector<Number>& y) -> char {
                Number max = 0;
                char imax = '!' - '0';
                for(char i=0; i<(char)y.size(); i++){
                  if(y[i]>max){
                    max = y[i];
                    imax = i;
                  } 
                } 
                return imax + '0';
              });
  std::cout << "Quatidade de amostras de treinamento: " << samples.size() << "\n";
  std::cout << "Quatidade de amostras de teste: " << vsamples.size() << "\n";

  network.addLayer(Layer(100,bipolarSigmoid));
  network.addLayer(Layer(100,bipolarSigmoid));
#else
  MLP network(samplesOR, samplesOR, linear,
              [](std::vector<Number>& y) -> char {
                return (y[0]>=0)? '1' : '0';
              });

  network.addLayer(Layer(2,bipolarSigmoid));
#endif

  
  auto start_time = std::chrono::high_resolution_clock::now(); 
  network.train();
  auto end_time = std::chrono::high_resolution_clock::now(); 
  
  std::chrono::duration<double> duration = end_time - start_time; 
  std::cout << "Training finished after " << duration.count() <<" seconds\n";
  
  network.updateMe(0);

  return 0;
}