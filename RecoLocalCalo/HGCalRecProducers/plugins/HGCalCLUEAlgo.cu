#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerTiles.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerTilesGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"




//GPU Add
#include <math.h>
#include <limits>
#include <iostream>





namespace HGCalRecAlgos{

  static const int maxNSeeds = 1024; 
  static const int maxNFollowers = 20; 
  static const int BufferSizePerSeed = 40; 


  __global__ void kernel_compute_histogram( HGCalLayerTilesGPU *d_hist, 
                                            CellsOnLayerPtr d_cells, 
                                            int numberOfCells
                                            ) 
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numberOfCells) {
      d_hist[0].fill(d_cells.x[idx], d_cells.y[idx], idx);
    }
  } //kernel

  __global__ void kernel_compute_density( HGCalLayerTilesGPU *d_hist, 
                                          CellsOnLayerPtr d_cells, 
                                          float delta_c, int numberOfCells
                                          ) 
  {
    int idxOne = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxOne < numberOfCells){

      // search box with histogram
      int xBinMin = d_hist[0].getXBin( d_cells.x[idxOne] - delta_c);
      int xBinMax = d_hist[0].getXBin( d_cells.x[idxOne] + delta_c);
      int yBinMin = d_hist[0].getYBin( d_cells.y[idxOne] - delta_c);
      int yBinMax = d_hist[0].getYBin( d_cells.y[idxOne] + delta_c);

      // loop over bins in search box
      for(int xBin = xBinMin; xBin < xBinMax+1; ++xBin) {
        for(int yBin = yBinMin; yBin < yBinMax+1; ++yBin) {
          int binIndex = d_hist[0].getGlobalBinByBin(xBin,yBin);
          int binSize  = d_hist[0][binIndex].size();

          // loop over bin contents
          for (int j = 0; j < binSize; j++) {
            int idxTwo = d_hist[0][binIndex][j];
            float distance = sqrt( (d_cells.x[idxOne]-d_cells.x[idxTwo])*(d_cells.x[idxOne]-d_cells.x[idxTwo]) + (d_cells.y[idxOne]-d_cells.y[idxTwo])*(d_cells.y[idxOne]-d_cells.y[idxTwo]));
            if(distance < delta_c) { 
              d_cells.rho[idxOne] += (idxOne == idxTwo ? 1. : 0.5) * d_cells.weight[idxTwo];
            }
          }
        }
      }
    }
  } //kernel


  __global__ void kernel_compute_distanceToHigher(HGCalLayerTilesGPU* d_hist, 
                                                  CellsOnLayerPtr d_cells, 
                                                  float delta_c, 
                                                  float outlierDeltaFactor_, 
                                                  int numberOfCells
                                                  ) 
  {
    int idxOne = blockIdx.x * blockDim.x + threadIdx.x;

    if (idxOne < numberOfCells){

      float idxOne_delta = std::numeric_limits<float>::max();
      int idxOne_nearestHigher = -1;

      // search box with histogram
      int xBinMin = d_hist[0].getXBin( d_cells.x[idxOne] - outlierDeltaFactor_*delta_c);
      int xBinMax = d_hist[0].getXBin( d_cells.x[idxOne] + outlierDeltaFactor_*delta_c);
      int yBinMin = d_hist[0].getYBin( d_cells.y[idxOne] - outlierDeltaFactor_*delta_c);
      int yBinMax = d_hist[0].getYBin( d_cells.y[idxOne] + outlierDeltaFactor_*delta_c);

      // loop over bins in search box
      for(int xBin = xBinMin; xBin < xBinMax+1; ++xBin) {
        for(int yBin = yBinMin; yBin < yBinMax+1; ++yBin) {
          int binIndex = d_hist[0].getGlobalBinByBin(xBin,yBin);
          int binSize  = d_hist[0][binIndex].size();

          // loop over bin contents
          for (int j = 0; j < binSize; j++) {
            int idxTwo = d_hist[0][binIndex][j];
            float distance = sqrt( (d_cells.x[idxOne]-d_cells.x[idxTwo])*(d_cells.x[idxOne]-d_cells.x[idxTwo]) + (d_cells.y[idxOne]-d_cells.y[idxTwo])*(d_cells.y[idxOne]-d_cells.y[idxTwo]));
            bool foundHigher = d_cells.rho[idxTwo] > d_cells.rho[idxOne];
            if(foundHigher && distance <= idxOne_delta) {
              // update i_delta
              idxOne_delta = distance;
              // update i_nearestHigher
              idxOne_nearestHigher = idxTwo;
            }
          }
        }
      } // finish looping over search box

      bool foundNearestHigherInSearchBox = (idxOne_nearestHigher != -1);
      // if i is not a seed or noise
      if (foundNearestHigherInSearchBox){
        // pass i_delta and i_nearestHigher to ith hit
        d_cells.delta[idxOne] = idxOne_delta;
        d_cells.nearestHigher[idxOne] = idxOne_nearestHigher;
      } else {
        // otherwise delta is garanteed to be larger outlierDeltaFactor_*delta_c
        // we can safely maximize delta to be maxDelta
        d_cells.delta[idxOne] = std::numeric_limits<float>::max();
        d_cells.nearestHigher[idxOne] = -1;
      }
    }
  } //kernel



  __global__ void kernel_find_clusters( GPU::VecArray<int,maxNSeeds>* d_seeds,
                                        GPU::VecArray<int,maxNFollowers>* d_followers,
                                        CellsOnLayerPtr d_cells,
                                        float delta_c, 
                                        float kappa_, 
                                        float outlierDeltaFactor_,
                                        int numberOfCells
                                        ) 
  {
    int idxOne = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idxOne < numberOfCells) {
      //printf("enter \n");
      float rho_c = kappa_ * d_cells.sigmaNoise[idxOne];

      // initialize clusterIndex
      d_cells.clusterIndex[idxOne] = -1;

      bool isSeed = (d_cells.delta[idxOne] > delta_c) && (d_cells.rho[idxOne] >= rho_c);
      bool isOutlier = (d_cells.delta[idxOne] > outlierDeltaFactor_*delta_c) && (d_cells.rho[idxOne] < rho_c);

      if (isSeed) {
        d_cells.isSeed[idxOne] = 1;
        d_seeds[0].push_back(idxOne);
      } else {
        if (!isOutlier) {
          int idxOne_NH = d_cells.nearestHigher[idxOne];
          d_followers[idxOne_NH].push_back(idxOne);  
        }
      }
    }
  } //kernel


  __global__ void kernel_assign_clusters( GPU::VecArray<int,maxNSeeds>* d_seeds, 
                                          GPU::VecArray<int,maxNFollowers>* d_followers,
                                          CellsOnLayerPtr d_cells,
                                          int* d_nClusters

                                          )
  {
    int idxCls = blockIdx.x * blockDim.x + threadIdx.x;
    int nClusters = d_seeds[0].size();
    d_nClusters[0] = nClusters;

    if (idxCls < nClusters){

      // buffer is "localStack"
      int buffer[BufferSizePerSeed];
      int bufferSize = 0;

      // asgine cluster to seed[idxCls]
      int idxThisSeed = d_seeds[0][idxCls];
      d_cells.clusterIndex[idxThisSeed] = idxCls;
      // push_back idThisSeed to buffer
      buffer[bufferSize] = idxThisSeed;
      bufferSize ++;

      // process all elements in buffer
      while (bufferSize>0){
        // get last element of buffer
        int idxEndOfBuffer = buffer[bufferSize-1];

        int temp_clusterIndex = d_cells.clusterIndex[idxEndOfBuffer];
        GPU::VecArray<int,maxNFollowers> temp_followers = d_followers[idxEndOfBuffer];
                
        // pop_back last element of buffer
        buffer[bufferSize-1] = 0;
        bufferSize--;

        // loop over followers of last element of buffer
        for( int j=0; j < temp_followers.size();j++ ){
          // pass id to follower
          d_cells.clusterIndex[temp_followers[j]] = temp_clusterIndex;

          // push_back follower to buffer
          buffer[bufferSize] = temp_followers[j];
          bufferSize++;
        }
      }
    }
  } //kernel





  int clueGPU(CellsOnLayer<float>& cellsOnLayer, float delta_c, float kappa_, float outlierDeltaFactor_) {

    int numberOfCells = cellsOnLayer.detid.size();
    std::cout << "numberOfCells = " << numberOfCells <<std::endl;
    
    CellsOnLayerPtr h_cells,d_cells;
    h_cells.initHost(cellsOnLayer);
    d_cells.initDevice(h_cells, numberOfCells);

    // define local variables : hist
    HGCalLayerTilesGPU *d_hist;
    cudaMalloc(&d_hist, sizeof(HGCalLayerTilesGPU));
    cudaMemset(d_hist, 0x00, sizeof(HGCalLayerTilesGPU));
    // define local variables :  seeds   
    GPU::VecArray<int,maxNSeeds> *d_seeds;
    cudaMalloc(&d_seeds, sizeof(GPU::VecArray<int,maxNSeeds>));
    cudaMemset(d_seeds, 0x00, sizeof(GPU::VecArray<int,maxNSeeds>));
    // define local variables :  followers
    GPU::VecArray<int,maxNFollowers> *d_followers;
    cudaMalloc(&d_followers, sizeof(GPU::VecArray<int,maxNFollowers>)*numberOfCells);
    cudaMemset(d_followers, 0x00, sizeof(GPU::VecArray<int,maxNFollowers>)*numberOfCells);
    // define local variables :  nclusters
    int numberOfClusters;
    int *h_nClusters, *d_nClusters;
    h_nClusters = &numberOfClusters;
    cudaMalloc(&d_nClusters, sizeof(int));
    cudaMemset(d_nClusters, 0x00, sizeof(int));


 
    // launch kernels
    const dim3 blockSize(128,1,1);
    const dim3 gridSize(ceil(numberOfCells/128.0),1,1);
    const dim3 gridSize_seeds(ceil(maxNSeeds/128.0),1,1);
    
    kernel_compute_histogram <<<gridSize,blockSize>>>(d_hist, d_cells, numberOfCells);
    kernel_compute_density <<<gridSize,blockSize>>>(d_hist, d_cells, delta_c, numberOfCells);
    kernel_compute_distanceToHigher <<<gridSize,blockSize>>>(d_hist, d_cells, delta_c, outlierDeltaFactor_, numberOfCells);
    kernel_find_clusters <<<gridSize,blockSize>>>(d_seeds, d_followers, d_cells, delta_c, kappa_, outlierDeltaFactor_, numberOfCells);
    kernel_assign_clusters <<<gridSize_seeds,blockSize>>>(d_seeds, d_followers, d_cells, d_nClusters);

    d_cells.cpyDToH(h_cells, numberOfCells);
    cudaMemcpy(h_nClusters, d_nClusters, sizeof(int), cudaMemcpyDeviceToHost);

    d_cells.freeDevice();
    cudaFree(d_hist);
    cudaFree(d_seeds);
    cudaFree(d_followers);
    cudaFree(d_nClusters);
    return numberOfClusters;

  }


}//namespace


