#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalCLUEAlgo.h"

// Geometry
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
//
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "tbb/task_arena.h"
#include "tbb/tbb.h"

//GPU Add
#include "RecoLocalCalo/HGCalRecProducers/interface/BinnerGPU.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/GPUHist2D.h"

#include <chrono>
#include <stdlib.h>     /* malloc, free, rand */
#include <iostream>
#include <fstream>


using namespace hgcal_clustering;

void HGCalCLUEAlgo::populate(const HGCRecHitCollection &hits) {
  // loop over all hits and create the Hexel structure, skip energies below ecut


  // for each layer and wafer calculate the thresholds (sigmaNoise and energy)
  // once
  computeThreshold();


  std::vector<bool> firstHit(2 * (maxlayer + 1), true);
  std::vector<unsigned int> layerCounters(2 * (maxlayer + 1),0);
    
  for (unsigned int i = 0; i < hits.size(); ++i) {
    const HGCRecHit &hgrh = hits[i];
    DetId detid = hgrh.detid();
    unsigned int layer = rhtools_.getLayerWithOffset(detid);
    float thickness = 0.f;
    // set sigmaNoise default value 1 to use kappa value directly in case of
    // sensor-independent thresholds
    float sigmaNoise = 1.f;

    thickness = rhtools_.getSiThickness(detid);
    int thickness_index = rhtools_.getSiThickIndex(detid);
    if (thickness_index == -1) thickness_index = 3;
    double storedThreshold = thresholds_[layer - 1][thickness_index];
    sigmaNoise = v_sigmaNoise_[layer - 1][thickness_index];

    if (hgrh.energy() < storedThreshold)
      continue;  // this sets the ZS threshold at ecut times the sigma noise
                  // for the sensor
    

    // map layers from positive endcap (z) to layer + maxlayer+1 to prevent
    // mixing up hits from different sides
    layer += int(rhtools_.zside(detid) > 0) * (maxlayer + 1);


    const GlobalPoint position(rhtools_.getPosition(detid));




    // ------------------------------------
    // MARK -- kdtree
    // ------------------------------------
    // here's were the KDNode is passed its dims arguments - note that these are
    // *copied* from the Hexel
    // determine whether this is a half-hexagon
    bool isHalf = rhtools_.isHalfCell(detid);
    points_[layer].emplace_back(Hexel(hgrh, detid, isHalf, sigmaNoise, thickness, &rhtools_),
                                position.x(), position.y());

    // for each layer, store the minimum and maximum x and y coordinates for the
    // KDTreeBox boundaries
    if (firstHit[layer]) {
      minpos_[layer][0] = position.x();
      minpos_[layer][1] = position.y();
      maxpos_[layer][0] = position.x();
      maxpos_[layer][1] = position.y();
      firstHit[layer] = false;
    } else {
      minpos_[layer][0] = std::min((float)position.x(), minpos_[layer][0]);
      minpos_[layer][1] = std::min((float)position.y(), minpos_[layer][1]);
      maxpos_[layer][0] = std::max((float)position.x(), maxpos_[layer][0]);
      maxpos_[layer][1] = std::max((float)position.y(), maxpos_[layer][1]);
    }

    // // ------------------------------------
    // // MARK -- bin cpu and gpu
    // // ------------------------------------
    // RecHitGPU hit;
    // hit.index = layerCounters[layer];
    // hit.x = position.x();
    // hit.y = position.y();
    // hit.eta = std::fabs(position.eta());
    // hit.phi = position.phi();
    // hit.weight = hgrh.energy();
    // hit.rho = 0.0;
    // hit.sigmaNoise = sigmaNoise;
    // recHitsGPU[layer].emplace_back(hit);;
    // layerCounters[layer]++;
    // thickness++;
  }  // end loop hits
}

// Create a vector of Hexels associated to one cluster from a collection of
// HGCalRecHits - this can be used directly to make the final cluster list -
// this method can be invoked multiple times for the same event with different
// input (reset should be called between events)
// start = std::chrono::high_resolution_clock::now();
// finish = std::chrono::high_resolution_clock::now();
// std::cout << "KDTree time ToT : " << (std::chrono::duration<double>(finish-start)).count() << " s \n" ;



void HGCalCLUEAlgo::makeClusters() {
  double timer0=0;
  double timer1=0;
  double timer2=0;
  double timer3=0;

  layerClustersPerLayer_.resize(2 * maxlayer + 2);
  // assign all hits in each layer to a cluster core
  tbb::this_task_arena::isolate([&] {
    tbb::parallel_for(size_t(0), size_t(2 * maxlayer + 2), [&](size_t i) {

      unsigned int actualLayer = i > maxlayer
                                ? (i - (maxlayer + 1))
                                : i;  // maps back from index used for KD trees to actual layer

      // ------------------------------------
      // MARK -- kdtree
      // ------------------------------------
      auto start = std::chrono::high_resolution_clock::now();
      KDTreeBox bounds(minpos_[i][0], maxpos_[i][0], minpos_[i][1], maxpos_[i][1]);
      KDTree hit_kdtree;
      hit_kdtree.build(points_[i], bounds);
      auto finish = std::chrono::high_resolution_clock::now();
      timer0 += (std::chrono::duration<double>(finish-start)).count();


      start = std::chrono::high_resolution_clock::now();
      double maxdensity = calculateLocalDensity(points_[i], hit_kdtree,actualLayer);
      finish = std::chrono::high_resolution_clock::now();
      timer1 += (std::chrono::duration<double>(finish-start)).count();

      start = std::chrono::high_resolution_clock::now();
      calculateDistanceToHigher(points_[i]);
      finish = std::chrono::high_resolution_clock::now();
      timer2 += (std::chrono::duration<double>(finish-start)).count();

      start = std::chrono::high_resolution_clock::now();
      findAndAssignClusters(points_[i], hit_kdtree, maxdensity, bounds, actualLayer,layerClustersPerLayer_[i]);
      finish = std::chrono::high_resolution_clock::now();
      timer3 += (std::chrono::duration<double>(finish-start)).count();

      
      // // ------------------------------------
      // // MARK -- Bin CPU
      // // ------------------------------------
      // // For each layer, assign all RecHits to a bin of a Histo2D
      // auto start = std::chrono::high_resolution_clock::now();
      // Histo2D histo(-250.0, 250.0, -250.0, 250.0);
      // for (unsigned int j=0; j<recHitsGPU[i].size(); j++) histo.fillBin(recHitsGPU[i][j].x, recHitsGPU[i][j].y, j);
      // auto finish = std::chrono::high_resolution_clock::now();
      // timer0 += (std::chrono::duration<double>(finish-start)).count();

      // start = std::chrono::high_resolution_clock::now();
      // calculateLocalDensity_BinCPU(histo, recHitsGPU[i], actualLayer);
      // finish = std::chrono::high_resolution_clock::now();
      // timer1 += (std::chrono::duration<double>(finish-start)).count();

      // start = std::chrono::high_resolution_clock::now();
      // calculateDistanceToHigher_BinCPU(histo, recHitsGPU[i], actualLayer);
      // finish = std::chrono::high_resolution_clock::now();
      // timer2 += (std::chrono::duration<double>(finish-start)).count();

      // start = std::chrono::high_resolution_clock::now();
      // findAndAssignClusters_BinCPU(recHitsGPU[i], actualLayer);
      // finish = std::chrono::high_resolution_clock::now();
      // timer3 += (std::chrono::duration<double>(finish-start)).count();


      // // ------------------------------------
      // // MARK -- Bin GPU
      // // ------------------------------------
      // HGCalRecAlgos::clue_BinGPU(recHitsGPU[i], actualLayer, vecDeltas_, kappa_, outlierDeltaFactor_);

      // // ------------------------------------
      // for (unsigned int j=0; j<recHitsGPU[i].size(); j++){
      //   auto temp = recHitsGPU[i][j];
      //   std::cout << "GPU RecHit N: " << j << " ("<<temp.x<<","<<temp.y<<")" << " | clusterIndex: " << temp.clusterIndex << " | Delta: " << temp.delta << " | NearestHigher: " << temp.nearestHigher << " | Density: " << temp.rho << " | rho_c: " << kappa_*temp.sigmaNoise << " |  nFollowers " << temp.followers.size() << std::endl;
      // }

    }); 
  });
  for(auto const& p: points_) { setDensity(p); }
  std::cout << "-- makeclus timer 0: " << timer0 << " s \n" ;
  std::cout << "-- makeclus timer 1: " << timer1 << " s \n" ;
  std::cout << "-- makeclus timer 2: " << timer2 << " s \n" ;
  std::cout << "-- makeclus timer 3: " << timer3 << " s \n" ;

}

std::vector<reco::BasicCluster> HGCalCLUEAlgo::getClusters(bool) {
  reco::CaloID caloID = reco::CaloID::DET_HGCAL_ENDCAP;
  std::vector<std::pair<DetId, float>> thisCluster;
  for (const auto &clsOnLayer : layerClustersPerLayer_) {
    int index = 0;
    for (const auto &cl : clsOnLayer) {
      double energy = 0;
      Point position;
      // Will save the maximum density hit of the cluster
      size_t rsmax = max_index(cl);
      position = calculatePosition(cl);  // energy-weighted position
      for (const auto &it : cl) {
        energy += it.data.weight;
        thisCluster.emplace_back(it.data.detid, 1.f);
      }
      if (verbosity_ < pINFO) {
        LogDebug("HGCalCLUEAlgo")
          << "******** NEW CLUSTER (HGCIA) ********"
          << "Index          " << index
          << "No. of cells = " << cl.size()
          << "     Energy     = " << energy
          << "     Phi        = " << position.phi()
          << "     Eta        = " << position.eta()
          << "*****************************" << std::endl;
      }
      clusters_v_.emplace_back(energy, position, caloID, thisCluster, algoId_);
      if (!clusters_v_.empty()) {
        clusters_v_.back().setSeed(cl[rsmax].data.detid);
      }
      thisCluster.clear();
      index++;
    }
  }
  return clusters_v_;
}

math::XYZPoint HGCalCLUEAlgo::calculatePosition(const std::vector<KDNode> &v) const {
  float total_weight = 0.f;
  float x = 0.f;
  float y = 0.f;

  unsigned int v_size = v.size();
  unsigned int maxEnergyIndex = 0;
  float maxEnergyValue = 0;

  // loop over hits in cluster candidate
  // determining the maximum energy hit
  for (unsigned int i = 0; i < v_size; i++) {
    if (v[i].data.weight > maxEnergyValue) {
      maxEnergyValue = v[i].data.weight;
      maxEnergyIndex = i;
    }
  }

  // Si cell or Scintillator. Used to set approach and parameters
  int thick = rhtools_.getSiThickIndex(v[maxEnergyIndex].data.detid);

  // for hits within positionDeltaRho_c_ from maximum energy hit
  // build up weight for energy-weighted position
  // and save corresponding hits indices
  std::vector<unsigned int> innerIndices;
  for (unsigned int i = 0; i < v_size; i++) {
    if (thick == -1 || distance2(v[i].data, v[maxEnergyIndex].data) < positionDeltaRho_c_[thick]) {
      innerIndices.push_back(i);

      float rhEnergy = v[i].data.weight;
      total_weight += rhEnergy;
      // just fill x, y for scintillator
      // for Si it is overwritten later anyway
      if (thick == -1) {
        x += v[i].data.x * rhEnergy;
        y += v[i].data.y * rhEnergy;
      }
    }
  }
  // just loop on reduced vector of interesting indices
  // to compute log weighting
  if (thick != -1 && total_weight != 0.) {  // Silicon case
    float total_weight_log = 0.f;
    float x_log = 0.f;
    float y_log = 0.f;
    for (auto idx : innerIndices) {
      float rhEnergy = v[idx].data.weight;
      if (rhEnergy == 0.) continue;
      float Wi = std::max(thresholdW0_[thick] + std::log(rhEnergy / total_weight), 0.);
      x_log += v[idx].data.x * Wi;
      y_log += v[idx].data.y * Wi;
      total_weight_log += Wi;
    }
    total_weight = total_weight_log;
    x = x_log;
    y = y_log;
  }

  if (total_weight != 0.) {
    auto inv_tot_weight = 1. / total_weight;
    return math::XYZPoint(x * inv_tot_weight, y * inv_tot_weight, v[maxEnergyIndex].data.z);
  }
  return math::XYZPoint(0, 0, 0);
}

double HGCalCLUEAlgo::calculateLocalDensity(std::vector<KDNode> &nd, KDTree &lp,
                                            const unsigned int layer) const {

  double maxdensity = 0.;
  float delta_c;  // maximum search distance (critical distance) for local
                  // density calculation
  if (layer <= lastLayerEE)
    delta_c = vecDeltas_[0];
  else if (layer <= lastLayerFH)
    delta_c = vecDeltas_[1];
  else
    delta_c = vecDeltas_[2];

  // for each node calculate local density rho and store it
  for (unsigned int i = 0; i < nd.size(); ++i) {
    // speec up search by looking within +/- delta_c window only
    KDTreeBox search_box(nd[i].dims[0] - delta_c, nd[i].dims[0] + delta_c, nd[i].dims[1] - delta_c,
                         nd[i].dims[1] + delta_c);
    std::vector<KDNode> found;
    lp.search(search_box, found);
    const unsigned int found_size = found.size();
    for (unsigned int j = 0; j < found_size; j++) {
      if (distance(nd[i].data, found[j].data) < delta_c) {
        nd[i].data.rho += (nd[i].data.detid == found[j].data.detid ? 1. : 0.5) * found[j].data.weight;
        //nd[i].data.rho += found[j].data.weight;

        maxdensity = std::max(maxdensity, nd[i].data.rho);
      }
    }  // end loop found
  }    // end loop nodes


  return maxdensity;
}

double HGCalCLUEAlgo::calculateDistanceToHigher(std::vector<KDNode> &nd) const {

  // sort vector of Hexels by decreasing local density
  std::vector<size_t> &&rs = sorted_indices(nd);

  double maxdensity = 0.0;
  int nearestHigher = -1;

  if (!rs.empty())
    maxdensity = nd[rs[0]].data.rho;
  else
    return maxdensity;  // there are no hits
  double dist2 = 0.;
  // start by setting delta for the highest density hit to
  // the most distant hit - this is a convention

  for (const auto &j : nd) {
    double tmp = distance2(nd[rs[0]].data, j.data);
    if (tmp > dist2) dist2 = tmp;
  }
  nd[rs[0]].data.delta = std::sqrt(dist2);
  nd[rs[0]].data.nearestHigher = nearestHigher;

  // now we save the largest distance as a starting point
  const double max_dist2 = dist2;
  const unsigned int nd_size = nd.size();

  for (unsigned int oi = 1; oi < nd_size; ++oi) {  // start from second-highest density
    dist2 = max_dist2;
    unsigned int i = rs[oi];
    // we only need to check up to oi since hits
    // are ordered by decreasing density
    // and all points coming BEFORE oi are guaranteed to have higher rho
    // and the ones AFTER to have lower rho
    for (unsigned int oj = 0; oj < oi; ++oj) {
      unsigned int j = rs[oj];
      double tmp = distance2(nd[i].data, nd[j].data);
      if (tmp <= dist2) {  // this "<=" instead of "<" addresses the (rare) case
                           // when there are only two hits
        dist2 = tmp;
        nearestHigher = j;
      }
    }
    nd[i].data.delta = std::sqrt(dist2);
    nd[i].data.nearestHigher = nearestHigher;  // this uses the original unsorted hitlist
  }


  return maxdensity;
}
int HGCalCLUEAlgo::findAndAssignClusters(std::vector<KDNode> &nd, KDTree &lp, double maxdensity,
                                         KDTreeBox &bounds, const unsigned int layer,
                                         std::vector<std::vector<KDNode>> &clustersOnLayer) const {
  // this is called once per layer and endcap...
  // so when filling the cluster temporary vector of Hexels we resize each time
  // by the number  of clusters found. This is always equal to the number of
  // cluster centers...
  
  unsigned int nClustersOnLayer = 0;
  float delta_c;  // critical distance
  if (layer <= lastLayerEE)
    delta_c = vecDeltas_[0];
  else if (layer <= lastLayerFH)
    delta_c = vecDeltas_[1];
  else
    delta_c = vecDeltas_[2];

  std::vector<size_t> rs = sorted_indices(nd);  // indices sorted by decreasing rho
  std::vector<size_t> ds = sort_by_delta(nd);   // sort in decreasing distance to higher

  const unsigned int nd_size = nd.size();
  for (unsigned int i = 0; i < nd_size; ++i) {
    if (nd[ds[i]].data.delta < delta_c) break;  // no more cluster centers to be looked at
    if (dependSensor_) {
      float rho_c = kappa_ * nd[ds[i]].data.sigmaNoise;
      if (nd[ds[i]].data.rho < rho_c) continue;  // set equal to kappa times noise threshold

    } else if (nd[ds[i]].data.rho * kappa_ < maxdensity)
      continue;

    nd[ds[i]].data.clusterIndex = nClustersOnLayer;
    if (verbosity_ < pINFO) {
      LogDebug("HGCalCLUEAlgo")
        << "Adding new cluster with index " << nClustersOnLayer
        << "Cluster center is hit " << ds[i] << std::endl;
    }
    nClustersOnLayer++;
  }

  // at this point nClustersOnLayer is equal to the number of cluster centers -
  // if it is zero we are  done
  if (nClustersOnLayer == 0) return nClustersOnLayer;

  // assign remaining points to clusters, using the nearestHigher set from
  // previous step (always set except
  // for top density hit that is skipped...)
  for (unsigned int oi = 1; oi < nd_size; ++oi) {
    unsigned int i = rs[oi];
    int ci = nd[i].data.clusterIndex;
    if (ci == -1 && nd[i].data.delta < 2. * delta_c) {
      nd[i].data.clusterIndex = nd[nd[i].data.nearestHigher].data.clusterIndex;
    }
  }

  // make room in the temporary cluster vector for the additional clusterIndex
  // clusters
  // from this layer
  if (verbosity_ < pINFO) {
    LogDebug("HGCalCLUEAlgo")
      << "resizing cluster vector by " << nClustersOnLayer << std::endl;
  }
  clustersOnLayer.resize(nClustersOnLayer);

  // Fill the cluster vector
  for (unsigned int i = 0; i < nd_size; ++i) {
    int ci = nd[i].data.clusterIndex;
    if (ci != -1) {
      clustersOnLayer[ci].push_back(nd[i]);
      if (verbosity_ < pINFO) {
        LogDebug("HGCalCLUEAlgo")
          << "Pushing hit " << i << " into cluster with index " << ci << std::endl;
      }
    }
  }

  // prepare the offset for the next layer if there is one
  if (verbosity_ < pINFO) {
    LogDebug("HGCalCLUEAlgo") << "moving cluster offset by " << nClustersOnLayer << std::endl;
  }

  return nClustersOnLayer;
}

void HGCalCLUEAlgo::computeThreshold() {
  // To support the TDR geometry and also the post-TDR one (v9 onwards), we
  // need to change the logic of the vectors containing signal to noise and
  // thresholds. The first 3 indices will keep on addressing the different
  // thicknesses of the Silicon detectors, while the last one, number 3 (the
  // fourth) will address the Scintillators. This change will support both
  // geometries at the same time.

  if (initialized_) return;  // only need to calculate thresholds once

  initialized_ = true;

  std::vector<double> dummy;
  const unsigned maxNumberOfThickIndices = 3;
  dummy.resize(maxNumberOfThickIndices + 1, 0);  // +1 to accomodate for the Scintillators
  thresholds_.resize(maxlayer, dummy);
  v_sigmaNoise_.resize(maxlayer, dummy);

  for (unsigned ilayer = 1; ilayer <= maxlayer; ++ilayer) {
    for (unsigned ithick = 0; ithick < maxNumberOfThickIndices; ++ithick) {
      float sigmaNoise = 0.001f * fcPerEle_ * nonAgedNoises_[ithick] * dEdXweights_[ilayer] /
                         (fcPerMip_[ithick] * thicknessCorrection_[ithick]);
      thresholds_[ilayer - 1][ithick] = sigmaNoise * ecut_;
      v_sigmaNoise_[ilayer - 1][ithick] = sigmaNoise;
    }
    float scintillators_sigmaNoise = 0.001f * noiseMip_ * dEdXweights_[ilayer];
    thresholds_[ilayer - 1][maxNumberOfThickIndices] = ecut_ * scintillators_sigmaNoise;
    v_sigmaNoise_[ilayer - 1][maxNumberOfThickIndices] = scintillators_sigmaNoise;
  }
}

void HGCalCLUEAlgo::setDensity(const std::vector<KDNode> &nd){

  // for each node store the computer local density
  for (auto &i : nd){
    density_[ i.data.detid ] =  i.data.rho ;
  }
}

Density HGCalCLUEAlgo::getDensity() {
  return density_;
}



double HGCalCLUEAlgo::calculateLocalDensity_BinCPU(Histo2D hist, LayerRecHitsGPU &hits, const unsigned int layer) const {

  double maxdensity = 0.0;
  float delta_c; // maximum search distance (critical distance) for local
                 // density calculation

  if (layer <= lastLayerEE)
    delta_c = vecDeltas_[0];
  else if (layer <= lastLayerFH)
    delta_c = vecDeltas_[1];
  else
    delta_c = vecDeltas_[2];
  

  for(unsigned int i = 0; i < hits.size(); i++) {

    //std::cout << "Calculate this hit at bin " << hist.getBinIdx(hits[i].x, hits[i].y) << std::endl;

    std::array<int,4> search_box = hist.searchBox(hits[i].x - delta_c, hits[i].x + delta_c, hits[i].y - delta_c, hits[i].y + delta_c);
    
    //std::cout << " and search_box is " << search_box[0] << " " << search_box[1] << " " << search_box[2] << " " << search_box[3] << std::endl;
    for(int xBin = search_box[0]; xBin < search_box[1]+1; ++xBin) {
      for(int yBin = search_box[2]; yBin < search_box[3]+1; ++yBin) {
        
        //int binId = hist.getBinIdx(hits[i].x,hits[i].y);
        int binId = hist.getBinIdx_byBins(xBin,yBin);
        //std::cout << "  neighbor bin"<< binId << std::endl;
        size_t binSize = hist.data_[binId].size();
        
        for (unsigned int j = 0; j < binSize; j++) {
          unsigned int idTwo = hist.data_[binId][j];
          if(distanceGPU(hits[i],hits[idTwo]) < delta_c) {
            hits[i].rho += (i == idTwo ? 1. : 0.5) * hits[idTwo].weight;
            maxdensity = std::max(maxdensity,hits[i].rho);
          }
        }
      }
    }    
  }

  return maxdensity;
}

double HGCalCLUEAlgo::calculateDistanceToHigher_BinCPU(Histo2D hist, LayerRecHitsGPU &hits, const unsigned int layer) const {

  float delta_c; 
  if (layer <= lastLayerEE)
    delta_c = vecDeltas_[0];
  else if (layer <= lastLayerFH)
    delta_c = vecDeltas_[1];
  else 
    delta_c = vecDeltas_[2];
  


  for(unsigned int i = 0; i < hits.size(); i++) {
    // initialize delta and nearest higer for i
    float i_delta = maxDelta_;
    int i_nearestHigher = -1;

    // get search box for ith hit
    // garrantee to cover "outlierDeltaFactor_*delta_c"
    std::array<int,4> search_box = hist.searchBox(hits[i].x - outlierDeltaFactor_*delta_c, hits[i].x + outlierDeltaFactor_*delta_c, hits[i].y - outlierDeltaFactor_*delta_c, hits[i].y + outlierDeltaFactor_*delta_c);
    
    // loop over all bins in the search box
    for(int xBin = search_box[0]; xBin < search_box[1]+1; ++xBin) {
      for(int yBin = search_box[2]; yBin < search_box[3]+1; ++yBin) {
        
        // get the id of this bin
        size_t binId = hist.getBinIdx_byBins(xBin,yBin);
        // get the size of this bin
        size_t binSize = hist.data_[binId].size();

        // loop over all hits in this bin
        for (unsigned int j = 0; j < binSize; j++) {
          int idTwo = hist.data_[binId][j];

          float dist = distanceGPU(hits[i],hits[idTwo]);
          bool foundHigher = hits[idTwo].rho > hits[i].rho;


          // if dist == i_delta, then last comer being the nearest higher
          if(foundHigher && dist <= i_delta) {

            // update i_delta
            i_delta = dist;
            // update i_nearestHigher
            i_nearestHigher = idTwo;
            
          }
        }
      }
    }

    bool foundNearestHigherInSearchBox = (i_delta != maxDelta_);
    //if (i_delta <= outlierDeltaFactor_*delta_c){
    if (foundNearestHigherInSearchBox){
      // pass i_delta and i_nearestHigher to ith hit
      hits[i].delta = i_delta;
      hits[i].nearestHigher = i_nearestHigher;
    } else {
      // otherwise delta is garanteed to be larger outlierDeltaFactor_*delta_c
      // we can safely maximize delta to be maxDelta
      hits[i].delta = maxDelta_;
      hits[i].nearestHigher = -1;
    }
  }

  return maxDelta_;
}


int HGCalCLUEAlgo::findAndAssignClusters_BinCPU( LayerRecHitsGPU &hits, const unsigned int layer ) const {
  // this is called once per layer and endcap...
  // so when filling the cluster temporary vector of Hexels we resize each time
  // by the number  of clusters found. This is always equal to the number of
  // cluster centers...

  //const int maxNFollower = 20;
  unsigned int nClustersOnLayer = 0;

  // buffer for index of hits, 
  // which has clusterIndex 
  // but have not pass clusterIndex to their followers
  std::queue<int> buffer; 

  // GPU::VecArray<int,maxNFollower> *followers;
  // followers = (GPU::VecArray<int,maxNFollower>*) malloc( hits.size() * sizeof(GPU::VecArray<int,maxNFollower>) );


  float delta_c; // critical distance
  if (layer <= lastLayerEE)
    delta_c = vecDeltas_[0];
  else if (layer <= lastLayerFH)
    delta_c = vecDeltas_[1];
  else
    delta_c = vecDeltas_[2];

  // find cluster seeds and outlier  
  for(unsigned int i = 0; i < hits.size(); i++) {

    float rho_c = kappa_ * hits[i].sigmaNoise;
    
    // initialize clusterIndex
    hits[i].clusterIndex = -1;

    bool isSeed = (hits[i].delta > delta_c) && (hits[i].rho >= rho_c);
    bool isOutlier = (hits[i].delta > outlierDeltaFactor_*delta_c) && (hits[i].rho < rho_c);

    if (isSeed) {
      // hits[i] is a seed
      hits[i].clusterIndex = nClustersOnLayer;
      nClustersOnLayer++;
      // add hits[i] into buffer
      buffer.push(i);
    
    } else if (!isOutlier) {
      hits[hits[i].nearestHigher].followers.push_back_unsafe(i);   
      //followers[ hits[i].nearestHigher ].push_back_unsafe(i); 
    } 
    
  }

  // hits in buffer, need to pass clusterIndex to their followers
  while (!buffer.empty()) {
    int frontOfBuffer = buffer.front();
    RecHitGPU thisHit = hits[frontOfBuffer];

    // auto thisHit_followers = hits[frontOfBuffer].followers;
    //GPU::VecArray<int,maxNFollower> thisHit_followers = followers[frontOfBuffer];

    buffer.pop();


    // loop over followers
    for( int j=0; j < thisHit.followers.size(); j++ ){
      // pass id to a follower
      hits[thisHit.followers[j]].clusterIndex = thisHit.clusterIndex;
      // push this follower to buffer
      buffer.push(thisHit.followers[j]);
    }
    
  }

  return nClustersOnLayer;
}