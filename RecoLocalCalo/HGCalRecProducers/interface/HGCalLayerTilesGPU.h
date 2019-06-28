#ifndef RecoLocalCalo_HGCalRecProducers_HGCalLayerTilesGPU
#define RecoLocalCalo_HGCalRecProducers_HGCalLayerTilesGPU

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
      xBin = min(xBin,nColumns_);
      xBin = max(xBin,0);
      // cannot use std:clap
      return xBin;
    }

    __host__ __device__
    int getYBin(float y) const {
      int yBin = (y-minY_)*rY_;
      yBin = min(yBin,nRows_);
      yBin = max(yBin,0);
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
    int4 searchBox(float xMin, float xMax, float yMin, float yMax){
      return int4{ getXBin(xMin), getXBin(xMax), getYBin(yMin), getYBin(yMax)};
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

  
#endif