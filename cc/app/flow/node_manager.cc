#include "app/flow/node_manager.h"

#include "library/timer/timer.h"

namespace app {
namespace flow {

NodeManager::NodeManager(const fs::path &car_path) :
 pc1_(new ktn::PointCloud()),
 pc2_(new ktn::PointCloud()),
 tn1_(new ktn::Tracklets()),
 tn2_(new ktn::Tracklets()),
 og1n_(new rtn::OccGrid()),
 og2n_(new rtn::OccGrid()),
 fin_(new fln::FlowImage()),
 cmn_(new fln::ClassificationMap()),
 dmn_(new fln::DistanceMap()),
 car_node_(new osgn::Car(car_path)) {
}

void NodeManager::SetViewer(const std::shared_ptr<vw::Viewer> &viewer) {
  viewer->AddChild(pc1_);
  viewer->AddChild(pc2_);
  viewer->AddChild(tn1_);
  viewer->AddChild(tn2_);
  viewer->AddChild(og1n_);
  viewer->AddChild(og2n_);
  viewer->AddChild(fin_);
  viewer->AddChild(cmn_);
  viewer->AddChild(dmn_);
  viewer->AddChild(car_node_);

  viewer_ = viewer;

  UpdateViewer();
}

void NodeManager::SetViewMode(int view_mode) {
  view_mode_ = view_mode;
  UpdateViewer();
}

void NodeManager::UpdateViewer() {
  if (view_mode_ == 1) {
    pc1_->Render(false);
    pc2_->Render(false);

    tn1_->Render(true);
    tn2_->Render(false);

    og1n_->Render(true);
    og2n_->Render(false);

    fin_->Render(true);
    cmn_->Render(false);

    //car_node_->Render(true);
  } else if (view_mode_ == 2) {
    pc1_->Render(false);
    pc2_->Render(false);

    tn1_->Render(false);
    tn2_->Render(true);

    og1n_->Render(false);
    og2n_->Render(true);

    fin_->Render(true);
    cmn_->Render(false);

    //car_node_->Render(true);
  } else if (view_mode_ == 3) {
    pc1_->Render(false);
    pc2_->Render(false);

    tn1_->Render(false);
    tn2_->Render(false);

    og1n_->Render(false);
    og2n_->Render(false);

    fin_->Render(false);
    cmn_->Render(true);

    //car_node_->Render(true);
  } else if (view_mode_ == 4) {
    pc1_->Render(true);
    pc2_->Render(false);

    tn1_->Render(false);
    tn2_->Render(false);

    og1n_->Render(false);
    og2n_->Render(false);

    fin_->Render(false);
    cmn_->Render(false);

    //car_node_->Render(true);
  } else if (view_mode_ == 5) {
    pc1_->Render(false);
    pc2_->Render(true);

    tn1_->Render(false);
    tn2_->Render(false);

    og1n_->Render(false);
    og2n_->Render(false);

    fin_->Render(false);
    cmn_->Render(false);

    //car_node_->Render(true);
  }
}

void NodeManager::Update(const fl::FlowProcessor &fp, const kt::VelodyneScan &scan1, const kt::VelodyneScan &scan2, kt::Tracklets *tracklets, int frame_num) {
  // Make sure we have a valid viewer
  if (!viewer_) {
    return;
  }

  viewer_->Lock();

  library::timer::Timer t;

  dmn_->Render(false);

  t.Start();
  pc1_->Update(scan1);
  pc2_->Update(scan2);
  printf("point cloud took %5.3f ms to render\n", t.GetMs());

  t.Start();
  tn1_->Update(tracklets, frame_num - 1);
  tn2_->Update(tracklets, frame_num);
  printf("Tracklets took %5.3f ms to render\n", t.GetMs());

  t.Start();
  og1n_->Update(fp.GetLastOccGrid1());
  og2n_->Update(fp.GetLastOccGrid2(), fp.GetClassificationMap());
  printf("Occ Grids took %5.3f ms to render\n", t.GetMs());

  t.Start();
  fin_->Update(fp.GetFlowImage());
  printf("Flow image took %5.3f ms to render\n", t.GetMs());

  t.Start();
  cmn_->Update(fp.GetClassificationMap());
  printf("Classification Map took %5.3f ms to render\n", t.GetMs());

  viewer_->Unlock();

  UpdateViewer();
}

void NodeManager::ShowDistanceMap(const fl::FlowProcessor &fp, double x, double y) {
  auto dm = fp.GetDistanceMap();

  viewer_->Lock();
  dmn_->Update(dm, x, y);
  viewer_->Unlock();

  dmn_->Render(true);
}

void NodeManager::ClearDistanceMap() {
  dmn_->Render(false);
}

} // flow
} // app
