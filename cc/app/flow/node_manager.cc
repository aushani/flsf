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
 fmn1_(new fln::FilterMap()),
 fmn2_(new fln::FilterMap()),
 fin_(new fln::FlowImage()),
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
  viewer->AddChild(fmn1_);
  viewer->AddChild(fmn2_);
  viewer->AddChild(fin_);
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

    fmn1_->Render(false);
    fmn2_->Render(false);

    fin_->Render(true);

    //car_node_->Render(true);
  } else if (view_mode_ == 2) {
    pc1_->Render(false);
    pc2_->Render(false);

    tn1_->Render(false);
    tn2_->Render(true);

    og1n_->Render(false);
    og2n_->Render(true);

    fmn1_->Render(false);
    fmn2_->Render(false);

    fin_->Render(true);

    //car_node_->Render(true);
  } else if (view_mode_ == 3) {
    pc1_->Render(true);
    pc2_->Render(false);

    tn1_->Render(true);
    tn2_->Render(false);

    og1n_->Render(false);
    og2n_->Render(false);

    fmn1_->Render(true);
    fmn2_->Render(false);

    fin_->Render(false);

    //car_node_->Render(true);
  } else if (view_mode_ == 4) {
    pc1_->Render(false);
    pc2_->Render(true);

    tn1_->Render(false);
    tn2_->Render(true);

    og1n_->Render(false);
    og2n_->Render(false);

    fmn1_->Render(false);
    fmn2_->Render(true);

    fin_->Render(false);

    //car_node_->Render(true);
  } else if (view_mode_ == 5) {
    pc1_->Render(true);
    pc2_->Render(false);

    tn1_->Render(false);
    tn2_->Render(false);

    og1n_->Render(false);
    og2n_->Render(false);

    fmn1_->Render(false);
    fmn2_->Render(false);

    fin_->Render(false);

    //car_node_->Render(true);
  } else if (view_mode_ == 6) {
    pc1_->Render(false);
    pc2_->Render(true);

    tn1_->Render(false);
    tn2_->Render(false);

    og1n_->Render(false);
    og2n_->Render(false);

    fmn1_->Render(false);
    fmn2_->Render(false);

    fin_->Render(false);

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
  og1n_->Update(fp.GetLastOccGrid1(), fp.GetBackgroundFilterMap1());
  og2n_->Update(fp.GetLastOccGrid2(), fp.GetBackgroundFilterMap2());;
  printf("Occ Grids took %5.3f ms to render\n", t.GetMs());

  t.Start();
  fin_->Update(fp.GetFlowImage());
  printf("Flow image took %5.3f ms to render\n", t.GetMs());

  //t.Start();
  //cmn_->Update(fp.GetClassificationMap());
  //printf("Classification Map took %5.3f ms to render\n", t.GetMs());

  t.Start();
  fmn1_->Update(fp.GetBackgroundFilterMap1());
  fmn2_->Update(fp.GetBackgroundFilterMap2());
  printf("Filter map took %5.3f ms to render\n", t.GetMs());

  viewer_->Unlock();

  UpdateViewer();
}

void NodeManager::ShowDistanceMap(const fl::FlowProcessor &fp, const kt::VelodyneScan &scan2, double x, double y) {
  const auto &dm = fp.GetDistanceMap();

  viewer_->Lock();
  dmn_->Update(dm, x, y);
  //pc2_->Update(scan2, dm, x, y);
  viewer_->Unlock();

  dmn_->Render(true);
}

void NodeManager::ClearDistanceMap() {
  dmn_->Render(false);
}

} // flow
} // app
