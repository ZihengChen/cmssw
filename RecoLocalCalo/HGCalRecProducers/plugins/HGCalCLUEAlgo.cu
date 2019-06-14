#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerTiles.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerTilesGPU.h"

#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"



//GPU Add
#include <math.h>
#include <limits>
#include <iostream>


namespace HGCalRecAlgos{
  // This has to be the same as cpu version
  static const unsigned int maxlayer = 52;
  static const unsigned int lastLayerEE = 28;
  static const unsigned int lastLayerFH = 40;

  static const int maxNSeeds = 4096; 
  static const int maxNFollowers = 20; 
  static const int BufferSizePerSeed = 20; 


  __device__ float getDeltaCFromLayer(int layer, float delta_c_EE, float delta_c_FH, float delta_c_BH){
    if (layer%maxlayer < lastLayerEE)
      return delta_c_EE;
    else if (layer%maxlayer < lastLayerFH)
      return delta_c_FH;
    else
      return delta_c_BH;
  }

  __global__ void kernel_compute_histogram( HGCalLayerTilesGPU *d_hist, 
                                            CellsOnLayerPtr d_cells, 
                                            int numberOfCells
                                            )
  {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numberOfCells) {
      int layer = d_cells.layer[idx];
      d_hist[layer].fill(d_cells.x[idx], d_cells.y[idx], idx);
    }
  } //kernel

  __global__ void kernel_compute_density( HGCalLayerTilesGPU *d_hist, 
                                          CellsOnLayerPtr d_cells, 
                                          float delta_c_EE, float delta_c_FH, float delta_c_BH,
                                          int numberOfCells
                                          ) 
  { 
    
    int idxOne = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxOne < numberOfCells){
      int layer = d_cells.layer[idxOne];
      float delta_c = getDeltaCFromLayer(layer, delta_c_EE, delta_c_FH, delta_c_BH);

      // search box with histogram
      int xBinMin = d_hist[layer].getXBin( d_cells.x[idxOne] - delta_c);
      int xBinMax = d_hist[layer].getXBin( d_cells.x[idxOne] + delta_c);
      int yBinMin = d_hist[layer].getYBin( d_cells.y[idxOne] - delta_c);
      int yBinMax = d_hist[layer].getYBin( d_cells.y[idxOne] + delta_c);

      // loop over bins in search box
      for(int xBin = xBinMin; xBin < xBinMax+1; ++xBin) {
        for(int yBin = yBinMin; yBin < yBinMax+1; ++yBin) {
          int binIndex = d_hist[layer].getGlobalBinByBin(xBin,yBin);
          int binSize  = d_hist[layer][binIndex].size();

          // loop over bin contents
          for (int j = 0; j < binSize; j++) {
            int idxTwo = d_hist[layer][binIndex][j];
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
                                                  float delta_c_EE, float delta_c_FH, float delta_c_BH,
                                                  float outlierDeltaFactor_, 
                                                  int numberOfCells
                                                  ) 
  {
    int idxOne = blockIdx.x * blockDim.x + threadIdx.x;

    if (idxOne < numberOfCells){
      int layer = d_cells.layer[idxOne];
      float delta_c = getDeltaCFromLayer(layer, delta_c_EE, delta_c_FH, delta_c_BH);


      float idxOne_delta = std::numeric_limits<float>::max();
      int idxOne_nearestHigher = -1;

      // search box with histogram
      int xBinMin = d_hist[layer].getXBin( d_cells.x[idxOne] - outlierDeltaFactor_*delta_c);
      int xBinMax = d_hist[layer].getXBin( d_cells.x[idxOne] + outlierDeltaFactor_*delta_c);
      int yBinMin = d_hist[layer].getYBin( d_cells.y[idxOne] - outlierDeltaFactor_*delta_c);
      int yBinMax = d_hist[layer].getYBin( d_cells.y[idxOne] + outlierDeltaFactor_*delta_c);

      // loop over bins in search box
      for(int xBin = xBinMin; xBin < xBinMax+1; ++xBin) {
        for(int yBin = yBinMin; yBin < yBinMax+1; ++yBin) {
          int binIndex = d_hist[layer].getGlobalBinByBin(xBin,yBin);
          int binSize  = d_hist[layer][binIndex].size();

          // loop over bin contents
          for (int j = 0; j < binSize; j++) {
            int idxTwo = d_hist[layer][binIndex][j];
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
                                        float delta_c_EE, float delta_c_FH, float delta_c_BH,
                                        float kappa_, 
                                        float outlierDeltaFactor_,
                                        int numberOfCells
                                        ) 
  {
    int idxOne = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idxOne < numberOfCells) {
      int layer = d_cells.layer[idxOne];
      float delta_c = getDeltaCFromLayer(layer, delta_c_EE, delta_c_FH, delta_c_BH);

      float rho_c = kappa_ * d_cells.sigmaNoise[idxOne];

      // initialize clusterIndex
      d_cells.clusterIndex[idxOne] = -1;

      bool isSeed = (d_cells.delta[idxOne] > delta_c) && (d_cells.rho[idxOne] >= rho_c);
      bool isOutlier = (d_cells.delta[idxOne] > outlierDeltaFactor_*delta_c) && (d_cells.rho[idxOne] < rho_c);

      if (isSeed) {
        d_cells.isSeed[idxOne] = 1;
        d_seeds[layer].push_back(idxOne);
      } else {
        if (!isOutlier) {
          int idxOne_NH = d_cells.nearestHigher[idxOne];
          d_followers[idxOne_NH].push_back(idxOne);  
        }
      }
    }
  } //kernel

  __global__ void kernel_get_n_clusters(GPU::VecArray<int,maxNSeeds>* d_seeds, int* d_nClusters)
  { 
    
    int idxLayer = threadIdx.x;
    d_nClusters[idxLayer] = d_seeds[idxLayer].size();
  }

  __global__ void kernel_assign_clusters( GPU::VecArray<int,maxNSeeds>* d_seeds, 
                                          GPU::VecArray<int,maxNFollowers>* d_followers,
                                          CellsOnLayerPtr d_cells,
                                          int * d_nClusters
                                          )
  {

    int idxLayer = blockIdx.x;
    int idxCls = blockIdx.y*blockDim.x + threadIdx.x;
    

    if (idxCls < d_nClusters[idxLayer]){

      // buffer is "localStack"
      int buffer[BufferSizePerSeed];
      int bufferSize = 0;

      // asgine cluster to seed[idxCls]
      int idxThisSeed = d_seeds[idxLayer][idxCls];
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





  void clueGPU(std::vector<CellsOnLayer<float>> & cells_, 
              std::vector<int> & numberOfClustersPerLayer_, 
              float delta_c_EE, 
              float delta_c_FH, 
              float delta_c_BH, 
              float kappa_,
              float outlierDeltaFactor_ 
              ) {
    const int numberOfLayers = cells_.size();

    //////////////////////////////////////////////
    // copy from cells to local SoA
    //////////////////////////////////////////////

    int indexLayerEnd[numberOfLayers];
    // populate local SoA
    CellsOnLayer<float> localSoA;
    for (int i=0; i < numberOfLayers; i++){
      localSoA.x.insert( localSoA.x.end(), cells_[i].x.begin(), cells_[i].x.end() );
      localSoA.y.insert( localSoA.y.end(), cells_[i].y.begin(), cells_[i].y.end() );
      localSoA.layer.insert( localSoA.layer.end(), cells_[i].layer.begin(), cells_[i].layer.end() );
      localSoA.weight.insert( localSoA.weight.end(), cells_[i].weight.begin(), cells_[i].weight.end() );
      localSoA.sigmaNoise.insert( localSoA.sigmaNoise.end(), cells_[i].sigmaNoise.begin(), cells_[i].sigmaNoise.end() );
      
      int numberOfCellsOnLayer = cells_[i].weight.size();
      if (i == 0){
        indexLayerEnd[i] = -1 + numberOfCellsOnLayer;
      } else {
        indexLayerEnd[i] = indexLayerEnd[i-1] + numberOfCellsOnLayer;
      }
    }  

    const int numberOfCells = indexLayerEnd[numberOfLayers-1] + 1;
    // prepare SoA
    localSoA.rho.resize(numberOfCells,0);
    localSoA.delta.resize(numberOfCells,9999999);
    localSoA.nearestHigher.resize(numberOfCells,-1);
    localSoA.clusterIndex.resize(numberOfCells,-1);
    localSoA.isSeed.resize(numberOfCells,0);

    //////////////////////////////////////////////
    // run on GPU
    //////////////////////////////////////////////

    CellsOnLayerPtr h_cells,d_cells;
    h_cells.initHost(localSoA);
    d_cells.initDevice(h_cells, numberOfCells);

    // define local variables : hist
    HGCalLayerTilesGPU *d_hist;
    cudaMalloc(&d_hist, sizeof(HGCalLayerTilesGPU) * numberOfLayers);
    cudaMemset(d_hist, 0x00, sizeof(HGCalLayerTilesGPU) * numberOfLayers);
    // define local variables :  seeds   
    GPU::VecArray<int,maxNSeeds> *d_seeds;
    cudaMalloc(&d_seeds, sizeof(GPU::VecArray<int,maxNSeeds>) * numberOfLayers);
    cudaMemset(d_seeds, 0x00, sizeof(GPU::VecArray<int,maxNSeeds>) * numberOfLayers);
    // define local variables :  followers
    GPU::VecArray<int,maxNFollowers> *d_followers;
    cudaMalloc(&d_followers, sizeof(GPU::VecArray<int,maxNFollowers>)*numberOfCells);
    cudaMemset(d_followers, 0x00, sizeof(GPU::VecArray<int,maxNFollowers>)*numberOfCells);


 
    // launch kernels
    const dim3 blockSize(64,1,1);
    const dim3 gridSize(ceil(numberOfCells/64.0),1,1);
    
    kernel_compute_histogram <<<gridSize,blockSize>>>(d_hist, d_cells, numberOfCells);
    kernel_compute_density <<<gridSize,blockSize>>>(d_hist, d_cells, delta_c_EE, delta_c_FH, delta_c_BH, numberOfCells);
    kernel_compute_distanceToHigher <<<gridSize,blockSize>>>(d_hist, d_cells, delta_c_EE, delta_c_FH, delta_c_BH, outlierDeltaFactor_, numberOfCells);
    kernel_find_clusters <<<gridSize,blockSize>>>(d_seeds, d_followers, d_cells, delta_c_EE, delta_c_FH, delta_c_BH, kappa_, outlierDeltaFactor_, numberOfCells);

    // define local variables :  nclusters
    int *h_nClusters, *d_nClusters;
    h_nClusters = numberOfClustersPerLayer_.data();
    cudaMalloc(&d_nClusters, sizeof(int)*numberOfLayers);
    cudaMemset(d_nClusters, 0x00, sizeof(int)*numberOfLayers);
    const dim3 nlayerBlockSize(numberOfLayers,1,1);
    const dim3 oneGridSize(1,1,1);
    kernel_get_n_clusters <<<oneGridSize,nlayerBlockSize>>>(d_seeds,d_nClusters);
    cudaMemcpy(h_nClusters, d_nClusters, sizeof(int)*numberOfLayers, cudaMemcpyDeviceToHost);
    // for (int i=0;i<numberOfLayers;i++) std::cout<<numberOfClustersPerLayer_[i] << std::endl;

    // assign clusters
    const dim3 BlockSize1024(1024,1);
    const dim3 nlayerGridSize(numberOfLayers,ceil(maxNSeeds/1024.0),1);
    
    kernel_assign_clusters <<<nlayerGridSize,BlockSize1024>>>(d_seeds, d_followers, d_cells, d_nClusters);

    // cuda free
    d_cells.cpyDToH(h_cells, numberOfCells);
    d_cells.freeDevice();
    cudaFree(d_hist);
    cudaFree(d_seeds);
    cudaFree(d_followers);
    cudaFree(d_nClusters);

    //////////////////////////////////////////////
    // copy from local SoA to cells 
    //////////////////////////////////////////////
    for (int i=0; i < numberOfLayers; i++){
      int numberOfCellsOnLayer = cells_[i].weight.size();
      int indexBegin = indexLayerEnd[i]+1 - numberOfCellsOnLayer;

      cells_[i].rho.resize(numberOfCellsOnLayer);
      cells_[i].delta.resize(numberOfCellsOnLayer);
      cells_[i].nearestHigher.resize(numberOfCellsOnLayer);
      cells_[i].clusterIndex.resize(numberOfCellsOnLayer);
      cells_[i].isSeed.resize(numberOfCellsOnLayer);

      memcpy(cells_[i].rho.data(), &localSoA.rho[indexBegin], sizeof(float)*numberOfCellsOnLayer);
      memcpy(cells_[i].delta.data(), &localSoA.delta[indexBegin], sizeof(float)*numberOfCellsOnLayer);
      memcpy(cells_[i].nearestHigher.data(), &localSoA.nearestHigher[indexBegin], sizeof(int)*numberOfCellsOnLayer);
      memcpy(cells_[i].clusterIndex.data(), &localSoA.clusterIndex[indexBegin], sizeof(int)*numberOfCellsOnLayer); 
      memcpy(cells_[i].isSeed.data(), &localSoA.isSeed[indexBegin], sizeof(int)*numberOfCellsOnLayer);

      // for (int j=0; j<numberOfCellsOnLayer; j++){
      //   int jj = indexBegin+j;
      //   cells_[i].rho.emplace_back(localSoA.rho[jj]);
      //   cells_[i].delta.emplace_back(localSoA.delta[jj]);
      //   cells_[i].nearestHigher.emplace_back(localSoA.nearestHigher[jj]);
      //   cells_[i].clusterIndex.emplace_back(localSoA.clusterIndex[jj]);
      //   cells_[i].isSeed.emplace_back(localSoA.isSeed[jj]);
      // }
    }

  }


}//namespace


