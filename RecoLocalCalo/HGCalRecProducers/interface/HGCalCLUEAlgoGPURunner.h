#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerTiles.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"

#include <cuda_runtime.h>
#include <cuda.h>


// This has to be the same as cpu version
static const unsigned int maxlayer = 52;
static const unsigned int lastLayerEE = 28;
static const unsigned int lastLayerFH = 40;

static const int maxNSeeds = 4096; 
static const int maxNFollowers = 20; 
static const int BufferSizePerSeed = 20; 


struct CellsOnLayerPtr
{
  float *x; 
  float *y ;
  int *layer ;
  float *weight ;
  float *sigmaNoise; 

  float *rho ; 
  float *delta; 
  int *nearestHigher;
  int *clusterIndex; 
  int *isSeed;
};



class ClueGPURunner{
    public:
        unsigned int numberOfCells = 0;
        unsigned int numberOfLayers = 0;

        CellsOnLayerPtr d_cells;
        GPU::VecArray<int,maxNSeeds> *d_seeds;
        GPU::VecArray<int,maxNFollowers> *d_followers;
        int *d_nClusters;
    
        ClueGPURunner(){
            init_device();
           
        }
        ~ClueGPURunner(){
            free_device();
        }

        void init_device(){
            unsigned int reserveNumberOfCells = 1000000;
            cudaMalloc(&d_cells.x, sizeof(float)*reserveNumberOfCells);
            cudaMalloc(&d_cells.y, sizeof(float)*reserveNumberOfCells);
            cudaMalloc(&d_cells.layer, sizeof(int)*reserveNumberOfCells);
            cudaMalloc(&d_cells.weight, sizeof(float)*reserveNumberOfCells);
            cudaMalloc(&d_cells.sigmaNoise, sizeof(float)*reserveNumberOfCells);
            cudaMalloc(&d_cells.rho, sizeof(float)*reserveNumberOfCells);
            cudaMalloc(&d_cells.delta, sizeof(float)*reserveNumberOfCells);
            cudaMalloc(&d_cells.nearestHigher, sizeof(int)*reserveNumberOfCells);
            cudaMalloc(&d_cells.clusterIndex, sizeof(int)*reserveNumberOfCells);
            cudaMalloc(&d_cells.isSeed, sizeof(int)*reserveNumberOfCells);

            unsigned int reserveNumberOfLayers = maxlayer*2 + 2;
            cudaMalloc(&d_seeds, sizeof(GPU::VecArray<int,maxNSeeds>) * reserveNumberOfLayers);
            cudaMalloc(&d_followers, sizeof(GPU::VecArray<int,maxNFollowers>)*reserveNumberOfCells);
            cudaMalloc(&d_nClusters, sizeof(int)*reserveNumberOfLayers);
        }

        void free_device(){
            cudaFree(d_cells.x);
            cudaFree(d_cells.y);
            cudaFree(d_cells.layer);
            cudaFree(d_cells.weight);
            cudaFree(d_cells.sigmaNoise);
            
            cudaFree(d_cells.rho);
            cudaFree(d_cells.delta);
            cudaFree(d_cells.nearestHigher);
            cudaFree(d_cells.clusterIndex);
            cudaFree(d_cells.isSeed);

            cudaFree(d_seeds);
            cudaFree(d_followers);
            cudaFree(d_nClusters);
        }

        // algorithm functions
        void clueGPU(std::vector<CellsOnLayer> &, std::vector<int> &, float, float, float, float, float);

        void assign_number_of_cells(unsigned int n){
            numberOfCells = n;
        }

        void assign_number_of_layers(unsigned int n){
            numberOfLayers = n;
        }


        void copy_todevice(CellsOnLayer& cellsOnLayer){
            cudaMemcpy(d_cells.x, cellsOnLayer.x.data(), sizeof(float)*numberOfCells, cudaMemcpyHostToDevice);
            cudaMemcpy(d_cells.y, cellsOnLayer.y.data(), sizeof(float)*numberOfCells, cudaMemcpyHostToDevice);
            cudaMemcpy(d_cells.layer, cellsOnLayer.layer.data(), sizeof(int)*numberOfCells, cudaMemcpyHostToDevice);
            cudaMemcpy(d_cells.weight, cellsOnLayer.weight.data(), sizeof(float)*numberOfCells, cudaMemcpyHostToDevice);
            cudaMemcpy(d_cells.sigmaNoise,cellsOnLayer.sigmaNoise.data(), sizeof(float)*numberOfCells, cudaMemcpyHostToDevice); 
        }

        void clear_set(){
            cudaMemset(d_cells.rho, 0x00, sizeof(float)*numberOfCells);
            cudaMemset(d_cells.delta, 0x00, sizeof(float)*numberOfCells);
            cudaMemset(d_cells.nearestHigher, 0x00, sizeof(int)*numberOfCells);
            cudaMemset(d_cells.clusterIndex, 0x00, sizeof(int)*numberOfCells);
            cudaMemset(d_cells.isSeed, 0x00, sizeof(int)*numberOfCells);

            cudaMemset(d_seeds, 0x00, sizeof(GPU::VecArray<int,maxNSeeds>) * numberOfLayers);
            cudaMemset(d_followers, 0x00, sizeof(GPU::VecArray<int,maxNFollowers>)*numberOfCells);
            cudaMemset(d_nClusters, 0x00, sizeof(int)*numberOfLayers);
        }

        void copy_tohost(CellsOnLayer& cellsOnLayer, std::vector<int> & numberOfClustersPerLayer_){
            cudaMemcpy(cellsOnLayer.rho.data(), d_cells.rho, sizeof(float)*numberOfCells, cudaMemcpyDeviceToHost);
            cudaMemcpy(cellsOnLayer.delta.data(), d_cells.delta, sizeof(float)*numberOfCells, cudaMemcpyDeviceToHost);
            cudaMemcpy(cellsOnLayer.nearestHigher.data(), d_cells.nearestHigher, sizeof(int)*numberOfCells, cudaMemcpyDeviceToHost);
            cudaMemcpy(cellsOnLayer.clusterIndex.data(), d_cells.clusterIndex, sizeof(int)*numberOfCells, cudaMemcpyDeviceToHost);
            cudaMemcpy(cellsOnLayer.isSeed.data(), d_cells.isSeed, sizeof(int)*numberOfCells, cudaMemcpyDeviceToHost);
            cudaMemcpy(numberOfClustersPerLayer_.data(), d_nClusters, sizeof(int)*numberOfLayers, cudaMemcpyDeviceToHost);
        }



        
};
