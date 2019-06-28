#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerTiles.h"
#include <cuda_runtime.h>
#include <cuda.h>

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
        CellsOnLayerPtr d_cells, h_cells;
        unsigned int numberOfCells = 1000000;
    
        ClueGPURunner(){
            init_device();
           
        }
        ~ClueGPURunner(){
            free_device();
        }

        void init_device(){
            unsigned int maxNumberOfCells = 1000000;
            cudaMalloc(&d_cells.x, sizeof(float)*maxNumberOfCells);
            cudaMalloc(&d_cells.y, sizeof(float)*maxNumberOfCells);
            cudaMalloc(&d_cells.layer, sizeof(int)*maxNumberOfCells);
            cudaMalloc(&d_cells.weight, sizeof(float)*maxNumberOfCells);
            cudaMalloc(&d_cells.sigmaNoise, sizeof(float)*maxNumberOfCells);
            cudaMalloc(&d_cells.rho, sizeof(float)*maxNumberOfCells);
            cudaMalloc(&d_cells.delta, sizeof(float)*maxNumberOfCells);
            cudaMalloc(&d_cells.nearestHigher, sizeof(int)*maxNumberOfCells);
            cudaMalloc(&d_cells.clusterIndex, sizeof(int)*maxNumberOfCells);
            cudaMalloc(&d_cells.isSeed, sizeof(int)*maxNumberOfCells);
        }


        void assign_cells_number(unsigned int n){
            numberOfCells = n;
        }


        void init_host(CellsOnLayer& cellsOnLayer){
            h_cells.x = cellsOnLayer.x.data();
            h_cells.y = cellsOnLayer.y.data();
            h_cells.layer = cellsOnLayer.layer.data();
            h_cells.weight = cellsOnLayer.weight.data();
            h_cells.sigmaNoise = cellsOnLayer.sigmaNoise.data();
            h_cells.rho = cellsOnLayer.rho.data();
            h_cells.delta = cellsOnLayer.delta.data();
            h_cells.nearestHigher = cellsOnLayer.nearestHigher.data();
            h_cells.clusterIndex = cellsOnLayer.clusterIndex.data();  
            h_cells.isSeed = cellsOnLayer.isSeed.data(); 
        }


        void copy_todevice(){
            cudaMemcpy(d_cells.x, h_cells.x, sizeof(float)*numberOfCells, cudaMemcpyHostToDevice);
            cudaMemcpy(d_cells.y, h_cells.y, sizeof(float)*numberOfCells, cudaMemcpyHostToDevice);
            cudaMemcpy(d_cells.layer, h_cells.layer, sizeof(int)*numberOfCells, cudaMemcpyHostToDevice);
            cudaMemcpy(d_cells.weight, h_cells.weight, sizeof(float)*numberOfCells, cudaMemcpyHostToDevice);
            cudaMemcpy(d_cells.sigmaNoise,h_cells.sigmaNoise, sizeof(float)*numberOfCells, cudaMemcpyHostToDevice); 
        }

        void clear_set(){
            cudaMemset(d_cells.rho, 0x00, sizeof(float)*numberOfCells);
            cudaMemset(d_cells.delta, 0x00, sizeof(float)*numberOfCells);
            cudaMemset(d_cells.nearestHigher, 0x00, sizeof(int)*numberOfCells);
            cudaMemset(d_cells.clusterIndex, 0x00, sizeof(int)*numberOfCells);
            cudaMemset(d_cells.isSeed, 0x00, sizeof(int)*numberOfCells);
        }

        void copy_tohost(){
            cudaMemcpy(h_cells.rho, d_cells.rho, sizeof(float)*numberOfCells, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_cells.delta, d_cells.delta, sizeof(float)*numberOfCells, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_cells.nearestHigher, d_cells.nearestHigher, sizeof(int)*numberOfCells, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_cells.clusterIndex, d_cells.clusterIndex, sizeof(int)*numberOfCells, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_cells.isSeed, d_cells.isSeed, sizeof(int)*numberOfCells, cudaMemcpyDeviceToHost);
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
        }

        void clueGPU(std::vector<CellsOnLayer> &, std::vector<int> &, float, float, float, float, float);
};
