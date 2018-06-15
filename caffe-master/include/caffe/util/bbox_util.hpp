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
	class V3BoxData {
	public:
		int label_;
		float score_;
		vector<float> box_;
	};

typedef map<int, vector<NormalizedBBox> > LabelBBox;

int int_index(vector<int> maskvalue, int bestn, int n);
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

bool BoxSortDecendScore(const V3BoxData& box1, const V3BoxData& box2);
void ApplyNms(const vector<V3BoxData>& boxes, vector<int>* idxes, float threshold);

template <typename Dtype>
vector<Dtype> get_dyolov3_box(const Dtype* x, vector<Dtype> biases, int index, int n, int i, int j, int w, int h)
{
	vector<Dtype> b;
	b.clear();
	b.push_back((i + x[index + 0]) / w);
	b.push_back((j + x[index + 1]) / h);
	b.push_back(exp(x[index + 2]) * biases[2 * n] / w);
	b.push_back(exp(x[index + 3]) * biases[2 * n + 1] / h);
	return b;
}

template <typename Dtype>
int yolo_num_detections(const Dtype* input_data, int side, int num_object, int num_class, float thresh)
{
	int sum = 0;
	for (int i = 0; i < side; i++)
	{
		for (int j = 0; j < side; j++)
		{
			for (int n = 0; n < num_object; n++)
			{
				int obj_index = int obj_index = (i*side + j)*basic_length*num_object + n*basic_length + 4;
				if (input_data[obj_index] > thresh)
				{
					sum += 1;
				}
			}
		}
	}
	return sum;
}

template <typename Dtype>
void GetYolov3GTBox(int side, int labelbaselength, const Dtype* label_data, map<int, vector<V3BoxData> >* gt_boxes) {
	int locations = side*side;
	for (int h = 0; h < side; h++)
	{
		for (int w = 0; w < side; w++)
		{
			V3BoxData gt_box;
			int gridindex = h*side + w;
			int label = static_cast<int>(label_data[locations * 2 + gridindex]);
			gt_box.label_ = label;
			gt_box.score_ = (float)gridindex / locations;
			int box_index = locations * 3 + gridindex * 4;
			for (int j = 0; j < 4; ++j) {
				gt_box.box_.push_back(label_data[box_index + j]);
			}
			if (gt_boxes->find(label) == gt_boxes->end()) {
				(*gt_boxes)[label] = vector<V3BoxData>(1, gt_box);
			}
			else {
				(*gt_boxes)[label].push_back(gt_box);
			}
		}
	}
}

template <typename Dtype>
void GetYolov3PredBoxes(int side, int num_object, int num_class, int basic_length, const Dtype* input_data,
	map<int, vector<V3BoxData> >* pred_boxes, vector<Dtype> biases, int score_type, float nms, float obj_threshold, float nms_threshold) {
	vector<V3BoxData> tmp_boxes;
	int pred_label = 0;
	Dtype max_prob = 0;
	for (int i = 0; i < side; i++)
	{
		for (int j = 0; j < side; j++)
		{
			for (int n = 0; n < num_object; n++)
			{
				V3BoxData pred_box;
				int box_index = (i*side + j)*basic_length*num_object + n*basic_length + 0;
				int obj_index = (i*side + j)*basic_length*num_object + n*basic_length + 4;
				float objectness = input_data[obj_index];
				if (objectness <= obj_threshold)
				{
					continue;
				}
				vector<Dtype> pred_bbox = get_dyolov3_box(input_data, biases, box_index, n, i, j, side, side);
				pred_box.box_.push_back(pred_bbox[0]);
				pred_box.box_.push_back(pred_bbox[1]);
				pred_box.box_.push_back(pred_bbox[2]);
				pred_box.box_.push_back(pred_bbox[3]);
				for (int c = 0; c < num_class; c++)
				{
					int class_index = (i*side + j)*basic_length*num_object + n*basic_length + 5 + c;
					Dtype prob = objectness*input_data[class_index];
					if (prob > max_prob)
					{
						max_prob = prob;
						pred_label = c;
					}
				}
				pred_box.label_ = pred_label;
				pred_box.score_ = max_prob;
				tmp_boxes.push_back(pred_box);
			}
		}
	}

	std::sort(tmp_boxes.begin(), tmp_boxes.end(), BoxSortDecendScore);
	vector<int> idxes;
	ApplyNms(tmp_boxes, &idxes, nms_threshold);
	for (int i = 0; i < idxes.size(); ++i) {
		V3BoxData box_data = tmp_boxes[idxes[i]];
		if (pred_boxes->find(box_data.label_) == pred_boxes->end()) {
			(*pred_boxes)[box_data.label_] = vector<V3BoxData>();
		}
		(*pred_boxes)[box_data.label_].push_back(box_data);
	}
}

template <typename Dtype>
void GetGtFormLabelsimple(int side, int labelbaselength, int maxboxes, const Dtype* label_data, vector<V3BoxData>* gt_boxes) {
	for (int i = 0; i < maxboxes; i++)
	{
		V3BoxData gt_box;
		int box_index = i*labelbaselength;
		gt_box.label_ = label_data[box_index + 4];
		Dtype x = label_data[box_index + 1];
		Dtype y = label_data[box_index + 2];
		if (!x) break;
		for (int j = 0; j < 4; ++j) {
			gt_box.box_.push_back(label_data[box_index + j]);
		}
		(*gt_boxes).push_back(gt_box);
	}
}

template <typename Dtype>
void GetGtFormLabelData(int side, int labelbaselength, const Dtype* label_data, vector<V3BoxData>* gt_boxes) {
	int locations = side*side;
	for (int h = 0; h < side; h++)
	{
		for (int w = 0; w < side; w++)
		{
			V3BoxData gt_box;
			int gridindex = h*side + w;
			int label = static_cast<int>(label_data[locations * 2 + gridindex]);
			bool isobj = label_data[locations + gridindex];
			gt_box.label_ = label;
			gt_box.score_ = (float)gridindex/locations;
			int box_index = locations * 3 + gridindex * 4;
			for (int j = 0; j < 4; ++j) {
				gt_box.box_.push_back(label_data[box_index + j]);
			}
			(*gt_boxes).push_back(gt_box);
		}
	}
}

template <typename Dtype>
void GetTainGtFormLabelData(int side, int labelbaselength, const Dtype* label_data, vector<V3BoxData>* gt_boxes) {
	for (int h = 0; h < side; h++)
	{
		for (int w = 0; w < side; w++)
		{
			V3BoxData gt_box;
			int grid_index = h*side + w;
			int label = static_cast<int>(label_data[locations * 2 + grid_index]);
			gt_box.label_ = label;
			gt_box.score_ = i;
			int box_index = locations * 3 + i * 4;
			for (int j = 0; j < 4; ++j) {
				gt_box.box_.push_back(label_data[box_index + j]);
			}
			if (gt_boxes->find(label) == gt_boxes->end()) {
				(*gt_boxes)[label] = vector<BoxData>(1, gt_box);
			}
			else {
				(*gt_boxes)[label].push_back(gt_box);
			}
		}
	}
}
                     
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
