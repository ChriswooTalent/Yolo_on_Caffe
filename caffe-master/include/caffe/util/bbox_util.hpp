#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#ifndef CAFFE_UTIL_BBOX_UTIL_H_
#define CAFFE_UTIL_BBOX_UTIL_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "caffe/caffe.hpp"

namespace caffe {

typedef map<int, vector<NormalizedBBox> > LabelBBox;

float BBoxSize(const NormalizedBBox& bbox, const bool normalized);
bool SortBBoxAscend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);
bool SortBBoxDescend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);
void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                   NormalizedBBox* intersect_bbox);
float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                     const bool normalized);
void CumSum(const vector<pair<float, int> >& pairs, vector<int>* cumsum);
void ComputeAP(const vector<pair<float, int> >& tp, int num_pos,
               const vector<pair<float, int> >& fp, string ap_version,
               vector<float>* prec, vector<float>* rec, float* ap);
                     
template <typename Dtype>
void setNormalizedBBox(NormalizedBBox& bbox, Dtype x, Dtype y, Dtype w, Dtype h)
{
  Dtype xmin = x - w/2.0;
  Dtype xmax = x + w/2.0;
  Dtype ymin = y - h/2.0;
  Dtype ymax = y + h/2.0;

  if (xmin < 0.0){
    xmin = 0.0;
  }
  if (xmax > 1.0){
    xmax = 1.0;
  }
  if (ymin < 0.0){
    ymin = 0.0;
  }
  if (ymax > 1.0){
    ymax = 1.0;
  }  
  bbox.set_xmin(xmin);
  bbox.set_ymin(ymin);
  bbox.set_xmax(xmax);
  bbox.set_ymax(ymax);
  float bbox_size = BBoxSize(bbox, true);
  bbox.set_size(bbox_size);
}



template <typename Dtype>
void GetDetectionResults(const Dtype* det_data, const int num_det,
      map<int, LabelBBox>* all_detections) {
  all_detections->clear();
  for (int i = 0; i < num_det; ++i) {
    int start_idx = i * 7;
    int item_id = det_data[start_idx];
    if (item_id == -1) {
      continue;
    }
    int label = det_data[start_idx + 1];
    NormalizedBBox bbox;
    Dtype x = det_data[start_idx + 3];
    Dtype y = det_data[start_idx + 4];
    Dtype w = det_data[start_idx + 5];
    Dtype h = det_data[start_idx + 6];

    setNormalizedBBox(bbox, x, y, w, h);
    bbox.set_score(det_data[start_idx + 2]); //confidence   
    (*all_detections)[item_id][label].push_back(bbox);
  }
}

template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int num_gt,
      map<int, LabelBBox >* all_gt_bboxes) {
  all_gt_bboxes->clear();
  int cnt = 0;
  for (int t = 0; t < 30; ++t){
    vector<Dtype> truth;
    int label = gt_data[t * 5];
    Dtype x = gt_data[t * 5 + 1];
    Dtype y = gt_data[t * 5 + 2];
    Dtype w = gt_data[t * 5 + 3];
    Dtype h = gt_data[t * 5 + 4];

    if (!w) break;
    cnt++;
    int item_id = 0;
    NormalizedBBox bbox;
    setNormalizedBBox(bbox, x, y, w, h);
    (*all_gt_bboxes)[item_id][label].push_back(bbox);
  }
}

template <typename Dtype>
Dtype Calc_rmse(const vector<Dtype>& box, const vector<Dtype>& truth) {
  return sqrt(pow(box[0]-truth[0], 2) +
              pow(box[1]-truth[1], 2) +
              pow(box[2]-truth[2], 2) +
              pow(box[3]-truth[3], 2));
}
template <typename Dtype>
Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
  Dtype l1 = x1 - w1/2;
  Dtype l2 = x2 - w2/2;
  Dtype left = l1 > l2 ? l1 : l2;
  Dtype r1 = x1 + w1/2;
  Dtype r2 = x2 + w2/2;
  Dtype right = r1 < r2 ? r1 : r2;
  return right - left;
}

template <typename Dtype>
Dtype Calc_iou(const vector<Dtype>& box, const vector<Dtype>& truth) {
  NormalizedBBox Bbox1, Bbox2;
  setNormalizedBBox(Bbox1, box[0], box[1], box[2], box[3]);
  setNormalizedBBox(Bbox2, truth[0], truth[1], truth[2], truth[3]);
  return JaccardOverlap(Bbox1, Bbox2, true);
}

template <typename Dtype>
void disp(Blob<Dtype>& swap)
{
  std::cout<<"#######################################"<<std::endl;
  for (int b = 0; b < swap.num(); ++b)
    for (int c = 0; c < swap.channels(); ++c)
      for (int h = 0; h < swap.height(); ++h)
      {
  	std::cout<<"[";
        for (int w = 0; w < swap.width(); ++w)
	{
	  std::cout<<swap.data_at(b,c,h,w)<<",";	
	}
	std::cout<<"]"<<std::endl;
      }
  return;
}


template <typename Dtype>
class PredictionResult{
  public:
    Dtype x;
    Dtype y;
    Dtype w;
    Dtype h;
    Dtype objScore;
    Dtype classScore;
    Dtype confidence;
    int classType;
};
template <typename Dtype>
void class_index_and_score(Dtype* input, int classes, PredictionResult<Dtype>& predict)
{
  Dtype sum = 0;
  Dtype large = input[0];
  int classIndex = 0;
  for (int i = 0; i < classes; ++i){
    if (input[i] > large)
      large = input[i];
  }
  for (int i = 0; i < classes; ++i){
    Dtype e = exp(input[i] - large);
    sum += e;
    input[i] = e;
  }
  
  for (int i = 0; i < classes; ++i){
    input[i] = input[i] / sum;   
  }
  large = input[0];
  classIndex = 0;

  for (int i = 0; i < classes; ++i){
    if (input[i] > large){
      large = input[i];
      classIndex = i;
    }
  }  
  predict.classType = classIndex ;
  predict.classScore = large;
}
template <typename Dtype>
void get_region_box(Dtype* x, PredictionResult<Dtype>& predict, vector<Dtype> biases, int n, int index, int i, int j, int w, int h){
  predict.x = (i + sigmoid(x[index + 0])) / w;
  predict.y = (j + sigmoid(x[index + 1])) / h;
  predict.w = exp(x[index + 2]) * biases[2*n] / w;
  predict.h = exp(x[index + 3]) * biases[2*n+1] / h;
}
template <typename Dtype>
void ApplyNms(vector< PredictionResult<Dtype> >& boxes, vector<int>& idxes, Dtype threshold) {
  map<int, int> idx_map;
  for (int i = 0; i < boxes.size() - 1; ++i) {
    if (idx_map.find(i) != idx_map.end()) {
      continue;
    }
    for (int j = i + 1; j < boxes.size(); ++j) {
      if (idx_map.find(j) != idx_map.end()) {
        continue;
      }
      NormalizedBBox Bbox1, Bbox2;
      setNormalizedBBox(Bbox1, boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h);
      setNormalizedBBox(Bbox2, boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h);

      float overlap = JaccardOverlap(Bbox1, Bbox2, true);

      if (overlap >= threshold) {
        idx_map[j] = 1;
      }
    }
  }
  for (int i = 0; i < boxes.size(); ++i) {
    if (idx_map.find(i) == idx_map.end()) {
      idxes.push_back(i);
    }
  }
}


template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

// Explicit initialization.
template bool SortScorePairDescend(const pair<float, int>& pair1,
                                   const pair<float, int>& pair2);
template bool SortScorePairDescend(const pair<float, pair<int, int> >& pair1,
                                   const pair<float, pair<int, int> >& pair2);
              

}  // namespace caffe

#endif  // CAFFE_UTIL_BBOX_UTIL_H_
