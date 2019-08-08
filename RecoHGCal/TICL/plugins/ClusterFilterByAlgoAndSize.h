// Authors: Marco Rovere - marco.rovere@cern.ch, Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 11/2018

#ifndef RecoHGCal_TICL_ClusterFilterByAlgoAndSize_H__
#define RecoHGCal_TICL_ClusterFilterByAlgoAndSize_H__

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "ClusterFilterBase.h"

#include <memory>
#include <utility>

// Filter clusters that belong to a specific algorithm
namespace ticl {
  class ClusterFilterByAlgoAndSize final : public ClusterFilterBase {
  public:
    ClusterFilterByAlgoAndSize(const edm::ParameterSet& ps)
        : ClusterFilterBase(ps),
          algo_number_(ps.getParameter<int>("algo_number")),
          min_cluster_size_(ps.getParameter<int>("min_cluster_size")),
          max_cluster_size_(ps.getParameter<int>("max_cluster_size")) {}
    ~ClusterFilterByAlgoAndSize() override{};

    std::unique_ptr<HgcalClusterFilterMask> filter(const std::vector<reco::CaloCluster>& layerClusters,
                                                   const HgcalClusterFilterMask& availableLayerClusters,
                                                   std::vector<float>& layerClustersMask) const override {
      auto filteredLayerClusters = std::make_unique<HgcalClusterFilterMask>();
      for (auto const& cl : availableLayerClusters) {
        
        if (layerClusters[cl.first].algo() == algo_number_ and
            layerClusters[cl.first].hitsAndFractions().size() <= max_cluster_size_ and
            (layerClusters[cl.first].hitsAndFractions().size() >= min_cluster_size_ or 
            (std::abs(layerClusters[cl.first].z())>430 )))  {
          filteredLayerClusters->emplace_back(cl);
        } else {
          layerClustersMask[cl.first] = 0.;
        }
      }
      return filteredLayerClusters;
    }

  private:
    int algo_number_;
    unsigned int min_cluster_size_;
    unsigned int max_cluster_size_;
  };
}  // namespace ticl

#endif
