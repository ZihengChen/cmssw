#ifndef RecoLocalCalo_HGCalRecProducers_HGCalCLUEAlgo_h
#define RecoLocalCalo_HGCalRecProducers_HGCalCLUEAlgo_h

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalClusteringAlgoBase.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalLayerTiles.h"


#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

// C/C++ headers
#include <set>
#include <string>
#include <vector>

using Density=hgcal_clustering::Density;

class HGCalCLUEAlgo : public HGCalClusteringAlgoBase {
 public:

 HGCalCLUEAlgo(const edm::ParameterSet& ps)
  : HGCalClusteringAlgoBase(
      (HGCalClusteringAlgoBase::VerbosityLevel)ps.getUntrackedParameter<unsigned int>("verbosity",3),
      reco::CaloCluster::undefined),
     thresholdW0_(ps.getParameter<std::vector<double> >("thresholdW0")),
     vecDeltas_(ps.getParameter<std::vector<double> >("deltac")),
     kappa_(ps.getParameter<double>("kappa")),
     ecut_(ps.getParameter<double>("ecut")),
     dependSensor_(ps.getParameter<bool>("dependSensor")),
     dEdXweights_(ps.getParameter<std::vector<double> >("dEdXweights")),
     thicknessCorrection_(ps.getParameter<std::vector<double> >("thicknessCorrection")),
     fcPerMip_(ps.getParameter<std::vector<double> >("fcPerMip")),
     fcPerEle_(ps.getParameter<double>("fcPerEle")),
     nonAgedNoises_(ps.getParameter<edm::ParameterSet>("noises").getParameter<std::vector<double> >("values")),
     noiseMip_(ps.getParameter<edm::ParameterSet>("noiseMip").getParameter<double>("noise_MIP")),
     initialized_(false),
     cellsCEE_(2*(maxlayer+1)),
     cellsCEH_(2*(maxlayer+1)),
     points_(2*(maxlayer+1)),
     minpos_(2*(maxlayer+1),{ {0.0f,0.0f} }),
     maxpos_(2*(maxlayer+1),{ {0.0f,0.0f} }) {}

  ~HGCalCLUEAlgo() override {}

  void populate(const HGCRecHitCollection &hits) override;
  void populateLayerTiles(const HGCRecHitCollection &hits);

  // this is the method that will start the clusterisation (it is possible to invoke this method
  // more than once - but make sure it is with different hit collections (or else use reset)

  void makeClusters() override;

  // this is the method to get the cluster collection out
  std::vector<reco::BasicCluster> getClusters(bool) override;

  void reset() override {
    clusters_v_.clear();
    for(auto& cl: numberOfClustersPerLayer_)
    {
      cl = 0;
    }

    for(auto& cells : cells_)
      cells.clear();
    
  }

  Density getDensity() override;

  void computeThreshold();

  static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    iDesc.add<std::vector<double>>("thresholdW0", {
        2.9,
        2.9,
        2.9
        });
    iDesc.add<std::vector<double>>("deltac", {
        1.3,
        1.3,
        5.0,
        });
    iDesc.add<bool>("dependSensor", true);
    iDesc.add<double>("ecut", 3.0);
    iDesc.add<double>("kappa", 9.0);
    iDesc.addUntracked<unsigned int>("verbosity", 3);
    iDesc.add<std::vector<double>>("dEdXweights",{});
    iDesc.add<std::vector<double>>("thicknessCorrection",{});
    iDesc.add<std::vector<double>>("fcPerMip",{});
    iDesc.add<double>("fcPerEle",0.0);
    edm::ParameterSetDescription descNestedNoises;
    descNestedNoises.add<std::vector<double> >("values", {});
    iDesc.add<edm::ParameterSetDescription>("noises", descNestedNoises);
    edm::ParameterSetDescription descNestedNoiseMIP;
    descNestedNoiseMIP.add<bool>("scaleByDose", false );
    iDesc.add<edm::ParameterSetDescription>("scaleByDose", descNestedNoiseMIP);
    descNestedNoiseMIP.add<std::string>("doseMap", "" );
    iDesc.add<edm::ParameterSetDescription>("doseMap", descNestedNoiseMIP);
    descNestedNoiseMIP.add<double>("noise_MIP", 1./100. );
    iDesc.add<edm::ParameterSetDescription>("noiseMip", descNestedNoiseMIP);
  }

  /// point in the space
  typedef math::XYZPoint Point;

 private:
  // To compute the cluster position
  std::vector<double> thresholdW0_;

  // The two parameters used to identify clusters
  std::vector<double> vecDeltas_;
  double kappa_;

  // The hit energy cutoff
  double ecut_;

  // For keeping the density per hit
  Density density_;

  // various parameters used for calculating the noise levels for a given sensor (and whether to use
  // them)
  bool dependSensor_;
  std::vector<double> dEdXweights_;
  std::vector<double> thicknessCorrection_;
  std::vector<double> fcPerMip_;
  double fcPerEle_;
  std::vector<double> nonAgedNoises_;
  double noiseMip_;
  std::vector<std::vector<double> > thresholds_;
  std::vector<std::vector<double> > v_sigmaNoise_;

  // initialization bool
  bool initialized_;

  float outlierDeltaFactor_ = 2.f;

  struct CellsOnLayer {
    std::vector<DetId> detid;
    std::vector<float> x; 
    std::vector<float> y;

    std::vector<float> weight; 
    std::vector<float> rho;

    std::vector<float> delta;
    std::vector<int> nearestHigher;
    std::vector<int> clusterIndex;
    std::vector<float> sigmaNoise;
    std::vector< std::vector <int> > followers;
    std::vector<bool> isSeed;

    void clear()
    {
      detid.clear();
      x.clear();
      y.clear();
      weight.clear();
      rho.clear();
      delta.clear();
      nearestHigher.clear();
      clusterIndex.clear();
      sigmaNoise.clear();
      followers.clear();
      isSeed.clear();
    }
  };


  template<class T>
  struct CellsOnLayer {
    std::vector<T> x; 
    std::vector<T> y;
    T z; 
    std::vector<bool> isHalfCell;

    std::vector<T> weight; 
    std::vector<DetId> detid;
    std::vector<T> rho;

    std::vector<T> delta;
    std::vector<int> nearestHigher;
    std::vector<int> clusterIndex;
    std::vector<float> sigmaNoise;
    std::vector<float> thickness;

  };

  //this are the tiles for the electromagnetic part
  std::vector<HGCalLayerTiles<hgcalTilesConstants::CEE>> layerTilesCEE_;
  //this are the tiles for the hadronic part
  std::vector<HGCalLayerTiles<hgcalTilesConstants::CEH>> layerTilesCEH_;

  std::vector<CellsOnLayer<double> > cellsCEE_;
  std::vector<CellsOnLayer<double> > cellsCEH_;




  typedef KDTreeLinkerAlgo<Hexel, 2> KDTree;
  typedef KDTreeNodeInfoT<Hexel, 2> KDNode;

  std::vector<std::vector<std::vector<KDNode> > > layerClustersPerLayer_;

  std::vector<size_t> sort_by_delta(const std::vector<KDNode> &v) const {
    std::vector<size_t> idx(v.size());
    std::iota(std::begin(idx), std::end(idx), 0);
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) { return v[i1].data.delta > v[i2].data.delta; });
    return idx;
  }

  std::vector<std::vector<KDNode> > points_;  // a vector of vectors of hexels, one for each layer
  //@@EM todo: the number of layers should be obtained programmatically - the range is 1-n instead
  //of 0-n-1...

  std::vector<std::array<float, 2> > minpos_;
  std::vector<std::array<float, 2> > maxpos_;

  // these functions should be in a helper class.
  inline double distance2(const Hexel &pt1, const Hexel &pt2) const {  // distance squared
    const double dx = pt1.x - pt2.x;
    const double dy = pt1.y - pt2.y;
    return (dx * dx + dy * dy);
  }  

  inline float distance(int cell1,
                         int cell2, int layerId) const {  // 2-d distance on the layer (x-y)
    return std::sqrt(distance2(cell1, cell2, layerId));
  }
  
  void prepareDataStructures(const unsigned int layerId);
  void calculateLocalDensity(const HGCalLayerTiles& lt, const unsigned int layerId, float delta_c);  // return max density
  void calculateDistanceToHigher(const HGCalLayerTiles& lt, const unsigned int layerId, float delta_c);
  int findAndAssignClusters(const unsigned int layerId, float delta_c);
  math::XYZPoint calculatePosition(const std::vector<int> &v, const unsigned int layerId) const;
  void setDensity(const unsigned int layerId);
};

#endif
