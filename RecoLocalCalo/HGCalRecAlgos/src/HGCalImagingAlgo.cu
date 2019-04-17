//#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalImagingAlgo.h"

//GPU Add
#include "RecoLocalCalo/HGCalRecAlgos/interface/BinnerGPU.h"
#include <math.h>

using namespace BinnerGPU;

namespace HGCalRecAlgos{


  static const unsigned int lastLayerEE = 28;
  static const unsigned int lastLayerFH = 40;

  __device__ double distance2GPU(const RecHitGPU pt1, const RecHitGPU pt2) {
    //distance squared
    const double dx = pt1.x - pt2.x;
    const double dy = pt1.y - pt2.y;
    return (dx*dx + dy*dy);
  } 



  __global__ void kernel_compute_density(Histo2D* theHist, RecHitGPU* theHits, float delta_c, int theHitsSize) {

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




  double calculateLocalDensity_BinGPU( const BinnerGPU::Histo2D& theHist, LayerRecHitsGPU& theHits, const unsigned int layer, std::vector<double> vecDeltas_) {

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
    // Call the kernel
    const dim3 blockSize(1024,1,1);
    const dim3 gridSize(1,1,1);
    kernel_compute_density <<<gridSize,blockSize>>>(dInputHist, dInputRecHits, delta_c, theHits.size());
    
    // Copy result back!/
    cudaMemcpy(hInputRecHits, dInputRecHits, sizeof(RecHitGPU)*theHits.size(), cudaMemcpyDeviceToHost);

    // Free all the memory
    cudaFree(dInputHist);
    cudaFree(dInputRecHits);
    
    // std::cout << "Inside GPU " << std::endl;
    for(unsigned int j = 0; j< theHits.size(); j++) {
      if (maxdensity < theHits[j].rho) {
        maxdensity = theHits[j].rho;
      }
    }

    return maxdensity;

  }//calcualteLocalDensity

}//namespace


// __global__ void kenrel_compute_distance_ToHigher(
//     RecHitGPU* nd,
//     size_t* rs, 
//     int* nearestHigher,
//     const double* max_dist2
// ){
//   size_t oi = threadIdx.x + 1;

//   {
//     double dist2 = *max_dist2;
//     unsigned int i = rs[oi];
//     // we only need to check up to oi since hits
//     // are ordered by decreasing density
//     // and all points coming BEFORE oi are guaranteed to have higher rho
//     // and the ones AFTER to have lower rho
//     for (unsigned int oj = 0; oj < oi; ++oj) {
//       unsigned int j = rs[oj];
//       double tmp = distance2GPU(nd[i], nd[j]);
//       if (tmp <= dist2) { // this "<=" instead of "<" addresses the (rare) case
//                           // when there are only two hits
//         dist2 = tmp;
//         *nearestHigher = j;
//       }
//     }
//     nd[i].delta = sqrt(dist2);
//     // this uses the original unsorted hitlist
//     nd[i].nearestHigher = *nearestHigher;
//   }
// }

// void launch_kenrel_compute_distance_ToHigher(
//   std::vector<RecHitGPU>& nd,
//   std::vector<size_t>& rs,
//   int& nearestHigher,
//   const double max_dist2
// ){
//   RecHitGPU* g_nd;
//   size_t* g_rs; 
//   int* g_nearestHigher;
//   double* g_max_dist2;

//   cudaMalloc(&g_nd, sizeof(RecHitGPU)*nd.size());
//   cudaMalloc(&g_rs, sizeof(size_t)*rs.size());
//   cudaMalloc(&g_nearestHigher,sizeof(int));
//   cudaMalloc(&g_max_dist2, sizeof(double));

//   cudaMemcpy(g_nd,            &nd[0],            sizeof(RecHitGPU)*nd.size(), cudaMemcpyHostToDevice);
//   cudaMemcpy(g_rs,            &rs[0],            sizeof(size_t)*rs.size(), cudaMemcpyHostToDevice);
//   cudaMemcpy(g_nearestHigher, &nearestHigher, sizeof(int), cudaMemcpyHostToDevice);
//   cudaMemcpy(g_max_dist2,     &max_dist2,     sizeof(double), cudaMemcpyHostToDevice);

//   const dim3 blockSize(nd.size()-1,1,1);
//   const dim3 gridSize(1,1,1);
//   kenrel_compute_distance_ToHigher <<<gridSize,blockSize>>>(g_nd, g_rs, g_nearestHigher,g_max_dist2);

//   cudaMemcpy(&nd[0],             g_nd,            sizeof(RecHitGPU)*nd.size(), cudaMemcpyDeviceToHost);
//   cudaMemcpy(&rs[0],             g_rs,            sizeof(size_t)*rs.size(), cudaMemcpyDeviceToHost);
//   cudaMemcpy(&nearestHigher,  g_nearestHigher, sizeof(int), cudaMemcpyDeviceToHost);

//   cudaFree(g_nd);
//   cudaFree(g_rs);
//   cudaFree(g_nearestHigher);
//   cudaFree(g_max_dist2);
// }

/*    size_t rechitLocation = blockIdx.x * blockDim.x + threadIdx.x;

    if(rechitLocation >= numRechits)
        return;

    float eta = dInputData[rechitLocation].eta;
    float phi = dInputData[rechitLocation].phi;
    unsigned int index = dInputData[rechitLocation].index;

    dOutputData->fillBinGPU(eta, phi, index);

}
*/
