// Author: Felice Pantaleo, Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 11/2018
#include <algorithm>
#include <set>
#include <vector>
#include <array>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PatternRecognitionbyCA.h"
#include "HGCGraph.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

using namespace ticl;

PatternRecognitionbyCA::PatternRecognitionbyCA(const edm::ParameterSet &conf) : PatternRecognitionAlgoBase(conf) {
  theGraph_ = std::make_unique<HGCGraph>();
  min_cos_theta_ = conf.getParameter<double>("min_cos_theta");
  min_cos_pointing_ = conf.getParameter<double>("min_cos_pointing");
  missing_layers_ = conf.getParameter<int>("missing_layers");
  min_clusters_per_ntuplet_ = conf.getParameter<int>("min_clusters_per_ntuplet");
  max_delta_time_ = conf.getParameter<double>("max_delta_time");
}

PatternRecognitionbyCA::~PatternRecognitionbyCA(){};

void PatternRecognitionbyCA::makeTracksters(const edm::Event &ev,
                                            const edm::EventSetup &es,
                                            const std::vector<reco::CaloCluster> &layerClusters,
                                            const std::vector<float> &mask,
                                            const edm::ValueMap<float> &layerClustersTime,
                                            const TICLLayerTiles &tiles,
                                            std::vector<Trackster> &result) {
  rhtools_.getEventSetup(es);

  theGraph_->setVerbosity(algo_verbosity_);
  theGraph_->clear();
  if (algo_verbosity_ > None) {
    LogDebug("HGCPatterRecoByCA") << "Making Tracksters with CA" << std::endl;
  }
  std::vector<HGCDoublet::HGCntuplet> foundNtuplets;
  std::vector<uint8_t> layer_cluster_usage(layerClusters.size(), 0);
  theGraph_->makeAndConnectDoublets(tiles,
                                    ticl::constants::nEtaBins,
                                    ticl::constants::nPhiBins,
                                    layerClusters,
                                    mask,
                                    layerClustersTime,
                                    2,
                                    2,
                                    min_cos_theta_,
                                    min_cos_pointing_,
                                    missing_layers_,
                                    rhtools_.lastLayerFH(),
                                    max_delta_time_);
  theGraph_->findNtuplets(foundNtuplets, min_clusters_per_ntuplet_);
  //#ifdef FP_DEBUG
  const auto &doublets = theGraph_->getAllDoublets();
  int tracksterId = 0;
  for (auto const &ntuplet : foundNtuplets) {
    std::set<unsigned int> effective_cluster_idx;
    for (auto const &doublet : ntuplet) {
      auto innerCluster = doublets[doublet].innerClusterId();
      auto outerCluster = doublets[doublet].outerClusterId();
      effective_cluster_idx.insert(innerCluster);
      effective_cluster_idx.insert(outerCluster);
      if (algo_verbosity_ > Advanced) {
        LogDebug("HGCPatterRecoByCA") << "New doublet " << doublet << " for trackster: " << result.size() << " InnerCl "
                                      << innerCluster << " " << layerClusters[innerCluster].x() << " "
                                      << layerClusters[innerCluster].y() << " " << layerClusters[innerCluster].z()
                                      << " OuterCl " << outerCluster << " " << layerClusters[outerCluster].x() << " "
                                      << layerClusters[outerCluster].y() << " " << layerClusters[outerCluster].z()
                                      << " " << tracksterId << std::endl;
      }
    }
    for (auto const i : effective_cluster_idx) {
      layer_cluster_usage[i]++;
      LogDebug("HGCPatterRecoByCA") << "LayerID: " << i << " count: " << (int)layer_cluster_usage[i] << std::endl;
    }
    // Put back indices, in the form of a Trackster, into the results vector
    Trackster tmp;
    tmp.vertices.reserve(effective_cluster_idx.size());
    tmp.vertex_multiplicity.resize(effective_cluster_idx.size(), 0);
    std::copy(std::begin(effective_cluster_idx), std::end(effective_cluster_idx), std::back_inserter(tmp.vertices));
    result.push_back(tmp);
    tracksterId++;
  }
  for (auto &trackster : result) {
    assert(trackster.vertices.size() <= trackster.vertex_multiplicity.size());
    for (size_t i = 0; i < trackster.vertices.size(); ++i) {
      trackster.vertex_multiplicity[i] = layer_cluster_usage[trackster.vertices[i]];
      LogDebug("HGCPatterRecoByCA") << "LayerID: " << trackster.vertices[i]
                                    << " count: " << (int)trackster.vertex_multiplicity[i] << std::endl;
    }
  }


  ///////////////////////////////////////////
  // MARK -- TensorFlow session
  ///////////////////////////////////////////

  unsigned numberOfTracksters = result.size();
  std::string pbFile = "/afs/cern.ch/user/z/zichen/public/TICL/CMSSW_11_0_X_2019-07-02-2300/src/RecoHGCal/TICL/plugins/CNN.pb";
  tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(pbFile);
  tensorflow::Session* session = tensorflow::createSession(graphDef);
  std::vector<tensorflow::Tensor> outputs;

  for(unsigned i = 0; i < numberOfTracksters; ++i)
  {   
    
    // save trackster features
    std::vector<std::vector<std::array<float, 3> >>  tracksterFeature;
    tracksterFeature.resize(53);

    float tracksterSumClEnergy = 0;
    for(unsigned j = 0; j < result[i].vertices.size(); j++)
    {
      const auto& lc = layerClusters[result[i].vertices[j]];
      std::array<float, 3> lcEtaPhiE {{ float(lc.eta()), float(lc.phi()), float(lc.energy()) }};
      const auto firstHitDetId = lc.hitsAndFractions()[0].first;
      int layer = rhtools_.getLayerWithOffset(firstHitDetId) - 1;
      tracksterFeature[layer].push_back(lcEtaPhiE);
      tracksterSumClEnergy += float(lc.energy());
    }

    // run tensorflow for trackster > 5 GeV
    if (tracksterSumClEnergy>5) {
      tensorflow::Tensor input(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 50, 10, 3})); // BxHxWxC
      // input.clear()
      // loop over layers to prepare input
      for(unsigned layer = 0; layer < 50; layer++)
      {
        // sort decreasing energy
        std::vector<std::array<float, 3>> layerData = tracksterFeature[layer];
        sort(layerData.begin(), layerData.end(), [](const std::array<float, 3> & a, const std::array<float, 3> & b) {return a[2] > b[2];});
        // push clusters to pixels of image
        for (unsigned icluster=0; icluster<10; icluster++) {
          if (icluster<layerData.size()){
            input.flat<float>().data()[3*10*layer+3*icluster+0] = layerData[icluster][0];
            input.flat<float>().data()[3*10*layer+3*icluster+1] = layerData[icluster][1];
            input.flat<float>().data()[3*10*layer+3*icluster+2] = layerData[icluster][2];
            std::cout << i << "," << layer << "," << layerData[icluster][0] << ", " << layerData[icluster][1] << ", " << layerData[icluster][2] << std::endl;
          }
          else {
            input.flat<float>().data()[3*10*layer+3*icluster+0] = float(0.0);
            input.flat<float>().data()[3*10*layer+3*icluster+1] = float(0.0);
            input.flat<float>().data()[3*10*layer+3*icluster+2] = float(0.0);
          }
        }
      }

      tensorflow::run(session, { { "input", input  }}, { "output/Softmax" }, &outputs);

      // print predictions
      auto prediction = outputs[0].matrix<float>();
      std::cout << "--" << i << "," << tracksterSumClEnergy << ","  <<prediction(0,0)<< "," << prediction(0,1) << ", " << prediction(0,2) << ", " << prediction(0,3) <<std::endl;
      // std::cout << "trackster energy " << tracksterSumClEnergy<<" GeV"<< std::endl;
      // std::cout << "------------------" << std::endl;
      // std::cout << "  - P electron: " << prediction(0,0) << std::endl;
      // std::cout << "  - P photon:   " << prediction(0,1) << std::endl;
      // std::cout << "  - P muon:     " << prediction(0,2) << std::endl;
      // std::cout << "  - P hadron:   " << prediction(0,3) << std::endl;
      // std::cout << "------------------" << std::endl;
      outputs.clear();
    } else {
      // std::cout << "trackster energy " << tracksterSumClEnergy<<" GeV,  not input to tensorflow" <<std::endl;
    }

      


  }
  // end of tensorflow session

}
