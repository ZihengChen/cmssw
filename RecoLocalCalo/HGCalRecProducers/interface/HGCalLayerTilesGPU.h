#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalLayerTilesGPU
#define RecoLocalCalo_HGCalRecAlgos_HGCalLayerTilesGPU

#include <memory>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalTilesConstants.h"


const float minX_   = hgcalTilesConstants::minX;
const float maxX_   = hgcalTilesConstants::maxX;
const float minY_   = hgcalTilesConstants::minY;
const float maxY_   = hgcalTilesConstants::maxY;
const int nColumns_ = hgcalTilesConstants::nColumns;
const int nRows_    = hgcalTilesConstants::nRows;

const float rX_ = nColumns_/(maxX_-minX_);
const float rY_ = nRows_/(maxY_-minY_);

class HGCalLayerTilesGPU {
  public:
    HGCalLayerTilesGPU()
    {}

    // overload the fill function on device
    __device__
    void fill(float x, float y, int i)
    {   
      tiles_[getGlobalBin(x,y)].push_back(i);
    }


    __host__ __device__
    int getXBin(float x) const {
      int xBin = (x-minX_)*rX_;
      // xBin = std::min(xBin,nColumns_);
      // xBin = std::max(xBin,0);
      // cannot use std:clap
      return xBin;
    }

    __host__ __device__
    int getYBin(float y) const {
      int yBin = (y-minY_)*rY_;
      // yBin = std::min(yBin,nRows_);
      // yBin = std::max(yBin,0);
      // cannot use std:clap
      return yBin;
    }

    __host__ __device__
    int getGlobalBin(float x, float y) const{
      return getXBin(x) + getYBin(y)*nColumns_;
    }

    __host__ __device__
    int getGlobalBinByBin(int xBin, int yBin) const {
      return xBin + yBin*nColumns_;
    }

    __host__ __device__
    std::array<int,4> searchBox(float xMin, float xMax, float yMin, float yMax){
      return std::array<int, 4>({{ getXBin(xMin), getXBin(xMax), getYBin(yMin), getYBin(yMax)}});
    }

    __host__ __device__
    void clear() {
      for(auto& t: tiles_) t.reset();
    }

    __host__ __device__
    GPU::VecArray<int, hgcalTilesConstants::maxTileDepth>& operator[](int globalBinId) {
      return tiles_[globalBinId];
    }

  private:
    GPU::VecArray<GPU::VecArray<int, hgcalTilesConstants::maxTileDepth>, hgcalTilesConstants::nColumns * hgcalTilesConstants::nRows > tiles_;

};


struct CellsOnLayerPtr
{
  float *x; 
  float *y ;
  float *weight ;
  float *sigmaNoise; 

  float *rho ; 
  float *delta; 
  int *nearestHigher;
  int *clusterIndex; 
  int *isSeed;


  void initHost(CellsOnLayer<float>& cellsOnLayer ){
    x = cellsOnLayer.x.data();
    y = cellsOnLayer.y.data();
    weight = cellsOnLayer.weight.data();
    sigmaNoise = cellsOnLayer.sigmaNoise.data();

    rho = cellsOnLayer.rho.data();
    delta = cellsOnLayer.delta.data();
    nearestHigher = cellsOnLayer.nearestHigher.data();
    clusterIndex = cellsOnLayer.clusterIndex.data();
    isSeed = cellsOnLayer.isSeed.data();
  }

  void initDevice(CellsOnLayerPtr h_cells, unsigned int numberOfCells){
    cudaMalloc(&x, sizeof(float)*numberOfCells);
    cudaMalloc(&y, sizeof(float)*numberOfCells);
    cudaMalloc(&weight, sizeof(float)*numberOfCells);
    cudaMalloc(&sigmaNoise, sizeof(float)*numberOfCells);
    cudaMemcpy(x, h_cells.x, sizeof(float)*numberOfCells, cudaMemcpyHostToDevice);
    cudaMemcpy(y, h_cells.y, sizeof(float)*numberOfCells, cudaMemcpyHostToDevice);
    cudaMemcpy(weight, h_cells.weight, sizeof(float)*numberOfCells, cudaMemcpyHostToDevice);
    cudaMemcpy(sigmaNoise, h_cells.sigmaNoise, sizeof(float)*numberOfCells, cudaMemcpyHostToDevice); 


    cudaMalloc(&rho, sizeof(float)*numberOfCells);
    cudaMemset(rho, 0x00, sizeof(float)*numberOfCells);
    cudaMalloc(&delta, sizeof(float)*numberOfCells);
    cudaMemset(delta, 0x00, sizeof(float)*numberOfCells);
    cudaMalloc(&nearestHigher, sizeof(int)*numberOfCells);
    cudaMemset(nearestHigher, 0x00, sizeof(int)*numberOfCells);
    cudaMalloc(&clusterIndex, sizeof(int)*numberOfCells);
    cudaMemset(clusterIndex, 0x00, sizeof(int)*numberOfCells);
    cudaMalloc(&isSeed, sizeof(int)*numberOfCells);
    cudaMemset(isSeed, 0x00, sizeof(int)*numberOfCells);
    cudaMemcpy(isSeed, h_cells.isSeed, sizeof(int)*numberOfCells, cudaMemcpyHostToDevice); 
  }

  void cpyDToH(CellsOnLayerPtr h_cells, unsigned int numberOfCells){
    cudaMemcpy(h_cells.rho, rho, sizeof(float)*numberOfCells, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cells.delta, delta, sizeof(float)*numberOfCells, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cells.nearestHigher, nearestHigher, sizeof(int)*numberOfCells, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cells.clusterIndex, clusterIndex, sizeof(int)*numberOfCells, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cells.isSeed, isSeed, sizeof(int)*numberOfCells, cudaMemcpyDeviceToHost);
  }

    


  void freeDevice(){
    cudaFree(x);
    cudaFree(y);
    cudaFree(weight);
    cudaFree(sigmaNoise);
    

    cudaFree(rho);
    cudaFree(delta);
    cudaFree(nearestHigher);
    cudaFree(clusterIndex);
    cudaFree(isSeed);
    
  }

};


  
#endif