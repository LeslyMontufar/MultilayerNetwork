#ifndef LAYER_H
#define LAYER_H

#include "definitions.h"
#include <random>


class Layer {
  public:
    size_t ny; 
    std::vector<Number> w;
    std::vector<Number> y;
    std::vector<Number> dyin;
    const std::vector<Number>* x;
    size_t nx;
    const act activation;
    std::vector<Number> dE_dz;
    std::vector<Number> dE_dx;
    std::vector<Number> dw;

    Layer(const size_t& nNeurons, const act& activation) : ny(nNeurons), activation(activation) {
      this->y.resize(nNeurons);
      this->dyin.resize(nNeurons);
      this->dE_dz.resize(nNeurons);
    }

    void initWeights(const std::vector<Number>& vx){
      nx = vx.size();
      x = &vx;
      w.resize(nx*ny+ny); 
      dw.resize(nx*ny+ny); 
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


#endif