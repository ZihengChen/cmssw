#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerTilesGPU.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalCLUEAlgoGPURunner.h"


//GPU Add
#include <math.h>
#include <limits>
#include <iostream>

// for timing
#include <chrono>
#include <ctime>



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
    double rho{0.};
    int layer = d_cells.layer[idxOne];
    float delta_c = getDeltaCFromLayer(layer, delta_c_EE, delta_c_FH, delta_c_BH);
    float xOne = d_cells.x[idxOne];
    float yOne = d_cells.y[idxOne];
    // search box with histogram
    int4 search_box = d_hist[layer].searchBox(xOne - delta_c, xOne + delta_c, yOne - delta_c, yOne + delta_c);

    // loop over bins in search box
    for(int xBin = search_box.x; xBin < search_box.y+1; ++xBin) {
      for(int yBin = search_box.z; yBin < search_box.w+1; ++yBin) {
        int binIndex = d_hist[layer].getGlobalBinByBin(xBin,yBin);
        int binSize  = d_hist[layer][binIndex].size();

        // loop over bin contents
        for (int j = 0; j < binSize; j++) {
          int idxTwo = d_hist[layer][binIndex][j];
          float xTwo = d_cells.x[idxTwo];
          float yTwo = d_cells.y[idxTwo];
          float distance = std::sqrt((xOne-xTwo)*(xOne-xTwo) + (yOne-yTwo)*(yOne-yTwo));
          if(distance < delta_c) { 
            rho += (idxOne == idxTwo ? 1. : 0.5) * d_cells.weight[idxTwo];              
          }
        }
      }
    }
    d_cells.rho[idxOne] = rho;
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
    float xOne = d_cells.x[idxOne];
    float yOne = d_cells.y[idxOne];
    float rhoOne = d_cells.rho[idxOne];

    // search box with histogram
    int4 search_box = d_hist[layer].searchBox(xOne - delta_c, xOne + delta_c, yOne - delta_c, yOne + delta_c);

    // loop over bins in search box
    for(int xBin = search_box.x; xBin < search_box.y+1; ++xBin) {
      for(int yBin = search_box.z; yBin < search_box.w+1; ++yBin) {
        int binIndex = d_hist[layer].getGlobalBinByBin(xBin,yBin);
        int binSize  = d_hist[layer][binIndex].size();

        // loop over bin contents
        for (int j = 0; j < binSize; j++) {
          int idxTwo = d_hist[layer][binIndex][j];
          float xTwo = d_cells.x[idxTwo];
          float yTwo = d_cells.y[idxTwo];
          float distance = std::sqrt((xOne-xTwo)*(xOne-xTwo) + (yOne-yTwo)*(yOne-yTwo));
          bool foundHigher = (d_cells.rho[idxTwo] > rhoOne) ;
          // in the rare case where rho is the same, use detid
          if (d_cells.rho[idxTwo] == rhoOne) {
            foundHigher = d_cells.detid[idxTwo] > d_cells.detid[idxOne];
          }
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

    float deltaOne = d_cells.delta[idxOne];
    float rhoOne = d_cells.rho[idxOne];

    bool isSeed = (deltaOne > delta_c) && (rhoOne >= rho_c);
    bool isOutlier = (deltaOne > outlierDeltaFactor_*delta_c) && (rhoOne < rho_c);

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



void ClueGPURunner::clueGPU(std::vector<CellsOnLayer> & cells_,
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
  // this is fast and takes 3~4 ms on a PU200 event
  //////////////////////////////////////////////
  // auto start1 = std::chrono::high_resolution_clock::now();

  int indexLayerEnd[numberOfLayers];
  // populate local SoA
  CellsOnLayer localSoA;
  for (int i=0; i < numberOfLayers; i++){
    localSoA.detid.insert( localSoA.detid.end(), cells_[i].detid.begin(), cells_[i].detid.end() ); 
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
  // auto finish1 = std::chrono::high_resolution_clock::now();

  //////////////////////////////////////////////
  // run on GPU
  //////////////////////////////////////////////
  // auto start2 = std::chrono::high_resolution_clock::now();

  assign_number_of_cells(numberOfCells);
  assign_number_of_layers(numberOfLayers);
  copy_todevice(localSoA);
  clear_set();

  // launch kernels
  const dim3 blockSize(1024,1,1);
  const dim3 gridSize(ceil(numberOfCells/1024.0),1,1);
  kernel_compute_histogram <<<gridSize,blockSize>>>(d_hist, d_cells, numberOfCells);
  kernel_compute_density <<<gridSize,blockSize>>>(d_hist, d_cells, delta_c_EE, delta_c_FH, delta_c_BH, numberOfCells);
  kernel_compute_distanceToHigher <<<gridSize,blockSize>>>(d_hist, d_cells, delta_c_EE, delta_c_FH, delta_c_BH, outlierDeltaFactor_, numberOfCells);
  kernel_find_clusters <<<gridSize,blockSize>>>(d_seeds, d_followers, d_cells, delta_c_EE, delta_c_FH, delta_c_BH, kappa_, outlierDeltaFactor_, numberOfCells);  
  
  const dim3 blockSize_nlayers(numberOfLayers,1,1);
  const dim3 gridSize_1(1,1,1);
  kernel_get_n_clusters <<<gridSize_1,blockSize_nlayers>>>(d_seeds,d_nClusters);

  const dim3 gridSize_nlayers(numberOfLayers,ceil(maxNSeeds/1024.0),1);
  kernel_assign_clusters <<<gridSize_nlayers,blockSize>>>(d_seeds, d_followers, d_cells, d_nClusters);

  copy_tohost(localSoA,numberOfClustersPerLayer_);
  // auto finish2 = std::chrono::high_resolution_clock::now();

  //////////////////////////////////////////////
  // copy from local SoA to cells 
  // this is fast and takes 1~2 ms on a PU200 event
  //////////////////////////////////////////////
  // auto start3 = std::chrono::high_resolution_clock::now();
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
  }

  // auto finish3 = std::chrono::high_resolution_clock::now();
  // std::cout << (std::chrono::duration<double>(finish1-start1)).count() << "," 
  //           << (std::chrono::duration<double>(finish2-start2)).count() << ","
  //           << (std::chrono::duration<double>(finish3-start3)).count() << ",";

}



