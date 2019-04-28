//#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalImagingAlgo.h"

//GPU Add
#include "RecoLocalCalo/HGCalRecAlgos/interface/BinnerGPU.h"
#include <math.h>

using namespace BinnerGPU;

namespace HGCalRecAlgos{


  static const unsigned int lastLayerEE = 28;
  static const unsigned int lastLayerFH = 40;
  static const float maxDelta = 2000.0; 
  static const unsigned int maxNSeeds = 200; 
  //static const unsigned int maxNFollowers = 20;


  __device__ double distance2GPU(const RecHitGPU pt1, const RecHitGPU pt2) {
    //distance squared
    const double dx = pt1.x - pt2.x;
    const double dy = pt1.y - pt2.y;
    return (dx*dx + dy*dy);
  } 



  __global__ void kernel_compute_density( Histo2D* theHist, RecHitGPU* theHits, 
                                          float delta_c, 
                                          int theHitsSize
                                          ) {

    size_t idOne = threadIdx.x;
    // int temp = theHist->getBinIdx_byBins(1,1);

    if (idOne < theHitsSize){


      int xBinMin = theHist->computeXBinIndex(std::max(float(theHits[idOne].x - delta_c), theHist->limits_[0]));
      int xBinMax = theHist->computeXBinIndex(std::min(float(theHits[idOne].x + delta_c), theHist->limits_[1]));
      int yBinMin = theHist->computeYBinIndex(std::max(float(theHits[idOne].y - delta_c), theHist->limits_[2]));
      int yBinMax = theHist->computeYBinIndex(std::min(float(theHits[idOne].y + delta_c), theHist->limits_[3]));

      // printf("%f %f %f %f \n", xBinMin, xBinMax, yBinMin, yBinMax);

      for(int xBin = xBinMin; xBin < xBinMax+1; ++xBin) {
        for(int yBin = yBinMin; yBin < yBinMax+1; ++yBin) {
          
          size_t binIndex = theHist->getBinIdx_byBins(xBin,yBin);
          size_t binSize  = theHist->data_[binIndex].size();


          for (unsigned int j = 0; j < binSize; j++) {
            int idTwo = (theHist->data_[binIndex])[j];

            double distance = sqrt(distance2GPU(theHits[idOne], theHits[idTwo]));

            if(distance < delta_c) {
              theHits[idOne].rho += theHits[idTwo].weight;
            }
          }
        }
      }
    }
  } //kernel


  __global__ void kernel_compute_distanceToHigher(Histo2D* theHist, RecHitGPU* theHits,
                                                  float delta_c, 
                                                  int theHitsSize, 
                                                  float outlierDeltaFactor_
                                                  ) {

    size_t idOne = threadIdx.x;
    // int temp = theHist->getBinIdx_byBins(1,1);

    if (idOne < theHitsSize){
      // initialize delta and nearest higer for i
      float i_delta = maxDelta;
      int i_nearestHigher = -1;

      // get search box for ith hit
      // garrantee to cover "outlierDeltaFactor_*delta_c"
      int xBinMin = theHist->computeXBinIndex(std::max(float(theHits[idOne].x - outlierDeltaFactor_*delta_c), theHist->limits_[0]));
      int xBinMax = theHist->computeXBinIndex(std::min(float(theHits[idOne].x + outlierDeltaFactor_*delta_c), theHist->limits_[1]));
      int yBinMin = theHist->computeYBinIndex(std::max(float(theHits[idOne].y - outlierDeltaFactor_*delta_c), theHist->limits_[2]));
      int yBinMax = theHist->computeYBinIndex(std::min(float(theHits[idOne].y + outlierDeltaFactor_*delta_c), theHist->limits_[3]));

      // loop over all bins in the search box
      for(int xBin = xBinMin; xBin < xBinMax+1; ++xBin) {
        for(int yBin = yBinMin; yBin < yBinMax+1; ++yBin) {
          
          // get the id of this bin
          size_t binIndex = theHist->getBinIdx_byBins(xBin,yBin);
          // get the size of this bin
          size_t binSize  = theHist->data_[binIndex].size();

          // loop over all hits in this bin
          for (unsigned int j = 0; j < binSize; j++) {
            int idTwo = (theHist->data_[binIndex])[j];

            double distance = sqrt(distance2GPU(theHits[idOne], theHits[idTwo]));
            bool foundHigher = theHits[idTwo].rho > theHits[idOne].rho;

            if(distance < i_delta && foundHigher) {
              // update i_delta
              i_delta = distance;
              // update i_nearestHigher
              i_nearestHigher = idTwo;
            }
          }
        }
      }
      
      bool foundNearestHigherInSearchBox = (i_delta != maxDelta);
      // if i is not a seed or noise
      if (foundNearestHigherInSearchBox){
        // pass i_delta and i_nearestHigher to ith hit
        theHits[idOne].delta = i_delta;
        theHits[idOne].nearestHigher = i_nearestHigher;
      } else {
        // otherwise delta is garanteed to be larger outlierDeltaFactor_*delta_c
        // we can safely maximize delta to be maxDelta
        theHits[idOne].delta = maxDelta;
        theHits[idOne].nearestHigher = -1;
      }
    }
  } //kernel


  __global__ void kernel_find_clusters( RecHitGPU* theHits, GPU::VecArray<int,maxNSeeds>* seeds,
                                        float delta_c, 
                                        int theHitsSize, 
                                        float kappa_, 
                                        float outlierDeltaFactor_
                                        ) {

    size_t idOne = threadIdx.x;

    if (idOne < theHitsSize){

      float rho_c = kappa_ * theHits[idOne].sigmaNoise;

      // initialize clusterIndex
      theHits[idOne].clusterIndex = -1;

      bool isSeed = (theHits[idOne].delta > delta_c) && (theHits[idOne].rho >= rho_c);
      bool isOutlier = (theHits[idOne].delta > outlierDeltaFactor_*delta_c) && (theHits[idOne].rho < rho_c);

      if (isSeed) {
        // decide a seed
        // push hits[idOne] into seeds
        seeds[0].push_back(idOne);
      } else {
        if (!isOutlier) {
          // if not a seed or outlier
          // register yourself as follower of your nearest higher
          int idNH = theHits[idOne].nearestHigher;
          theHits[idNH].followers.push_back(idOne);  
        }
      }
    }
  } //kernel


  __global__ void kernel_assign_clusters( RecHitGPU* theHits, GPU::VecArray<int,maxNSeeds>* seeds ) {

    unsigned int idOne = threadIdx.x;
    unsigned int nClusters = seeds[0].size();

    if (idOne < nClusters){

      // buffer for index of hits, 
      // which has clusterIndex 
      // but have not pass clusterIndex to their followers

      // GPU::VecArray<int,maxNSeeds> buffer;
      // buffer.reset();

      // // asgine cluster to seed[idOne]
      // int idThisSeed = seeds[0][idOne];
      // theHits[idThisSeed].clusterIndex = idOne;
      // buffer.push_back(idThisSeed);

      // // starting from hit 
      // while(!buffer.empty()){

      //   // last element of buffer
      //   int idEndOfBuffer = buffer[buffer.size()-1];
      //   RecHitGPU thisHit = theHits[ idEndOfBuffer ];
                
      //   // pop last element of buffer
      //   buffer.pop_back();
        

      //   // loop over followers
      //   for( int j=0; j < thisHit.followers.size();j++ ){
      //     // pass id to follower
      //     theHits[thisHit.followers[j]].clusterIndex = thisHit.clusterIndex;
      //     // push follower to buffer
      //     buffer.push_back(thisHit.followers[j]);
      //   }
      // }

      int buffer[maxNSeeds];
      int bufferSize = 0;

      // asgine cluster to seed[idOne]
      int idThisSeed = seeds[0][idOne];
      theHits[idThisSeed].clusterIndex = idOne;
      // push_back idThisSeed to buffer
      buffer[bufferSize] = idThisSeed;
      bufferSize ++;

      // process all elements in buffer
      while (bufferSize>0){
        // get last element of buffer
        int idEndOfBuffer = buffer[bufferSize-1];
        RecHitGPU thisHit = theHits[ idEndOfBuffer ];
                
        // pop_back last element of buffer
        buffer[bufferSize-1] = 0;
        bufferSize--;

        // loop over followers of last element of buffer
        for( int j=0; j < thisHit.followers.size();j++ ){
          // pass id to follower
          theHits[thisHit.followers[j]].clusterIndex = thisHit.clusterIndex;
          
          // push_back follower to buffer
          buffer[bufferSize] = thisHit.followers[j];
          bufferSize++;
        }
      }








    }
  } //kernel





  double clue_BinGPU( const BinnerGPU::Histo2D& theHist, LayerRecHitsGPU& theHits, 
                      const unsigned int layer,
                      std::vector<double> vecDeltas_, 
                      float kappa_, 
                      float outlierDeltaFactor_) {

    double maxdensity = 0.;
    float delta_c;
    

    // maximum search distance (critical distance) for local density calculation
    
    if (layer <= lastLayerEE)
      delta_c = vecDeltas_[0];
    else if (layer <= lastLayerFH)
      delta_c = vecDeltas_[1];
    else
      delta_c = vecDeltas_[2];

    RecHitGPU *hInputRecHits;
    hInputRecHits = theHits.data();
    

    Histo2D *dInputHist;
    RecHitGPU *dInputRecHits;  // make input hits for GPU


    int numBins = theHist.data_.size();

    cudaMalloc(&dInputHist, sizeof(Histo2D));
    cudaMalloc(&dInputRecHits, sizeof(RecHitGPU)*theHits.size());

    cudaMemcpy(dInputHist, &theHist, sizeof(Histo2D), cudaMemcpyHostToDevice);
    cudaMemcpy(dInputRecHits, hInputRecHits, sizeof(RecHitGPU)*theHits.size(), cudaMemcpyHostToDevice);
    std::cout<< "h->d sizeof(RecHitGPU) is " << sizeof(RecHitGPU) <<std::endl;


    // define GPU vecArray
    GPU::VecArray<int,maxNSeeds> *dSeeds;
    //GPU::VecArray<int,maxNFollowers> *dFollowers;
    std::cout<< "GPU::VecArray<int,maxNSeeds> is "<<sizeof(GPU::VecArray<int,maxNSeeds>) <<  std::endl;
    cudaMalloc(&dSeeds, sizeof(GPU::VecArray<int,maxNSeeds>));
    cudaMemset(dSeeds, 0x00, sizeof(GPU::VecArray<int,maxNSeeds>));
    //cudaMalloc(&dFollowers, sizeof(GPU::VecArray<int,maxNFollowers>) * theHits.size());
    
    
    // Call the kernel
    const dim3 blockSize(1024,1,1);
    const dim3 gridSize(1,1,1);

    kernel_compute_density <<<gridSize,blockSize>>>(dInputHist, dInputRecHits, 
                                                    delta_c, 
                                                    theHits.size()
                                                    );

    kernel_compute_distanceToHigher <<<gridSize,blockSize>>>( dInputHist, dInputRecHits, 
                                                              delta_c, 
                                                              theHits.size(), 
                                                              outlierDeltaFactor_
                                                              );

    kernel_find_clusters <<<gridSize,blockSize>>>(dInputRecHits, dSeeds,
                                                  delta_c, 
                                                  theHits.size(), 
                                                  kappa_, 
                                                  outlierDeltaFactor_
                                                  );
                                                  
    kernel_assign_clusters <<<gridSize,blockSize>>>(dInputRecHits, dSeeds);

    // Copy result back!/
    cudaMemcpy(hInputRecHits, dInputRecHits, sizeof(RecHitGPU)*theHits.size(), cudaMemcpyDeviceToHost);
    std::cout<< "d->h sizeof(RecHitGPU) is " << sizeof(RecHitGPU) <<std::endl;


    // Free all the memory
    cudaFree(dInputHist);
    cudaFree(dInputRecHits);
    cudaFree(dSeeds);
    
    for(unsigned int j = 0; j< theHits.size(); j++) {
      if (maxdensity < theHits[j].rho) {
        maxdensity = theHits[j].rho;
      }
    }

    return maxdensity;

  }




}//namespace

