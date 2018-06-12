#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include "caffe/layers/yolov3_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/detection_evaluate_layer.hpp"
#include "caffe/util/bbox_util.hpp"

int iter_yolov3 = 0;

namespace caffe {
template <typename Dtype>
vector<Dtype> get_yolov3_box(Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int w, int h)
{
  vector<Dtype> b;
  b.clear();
  b.push_back((i + x[index + 0]) / w);
  b.push_back((j + x[index + 1]) / h);
  b.push_back(exp(x[index + 2]) * biases[2*n] / w);
  b.push_back(exp(x[index + 3]) * biases[2*n+1] / h);
  return b;
}

template <typename Dtype>
Dtype delta_yolov3region_box(vector<Dtype> truth, Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int w, int h, Dtype* delta, float scale){
  vector<Dtype> pred;
  pred.clear();
  pred = get_yolov3_box(x, biases, n, index, i, j, w, h);
        
  float iou = Calc_iou(pred, truth);
  //LOG(INFO) << pred[0] << "," << pred[1] << "," << pred[2] << "," << pred[3] << ";"<< truth[0] << "," << truth[1] << "," << truth[2] << "," << truth[3];
  float tx = truth[0] * w - i; //0.5
  float ty = truth[1] * h - j; //0.5
  float tw = log(truth[2] * w / biases[2*n]); //truth[2]=biases/w tw = 0
  float th = log(truth[3] * h / biases[2*n + 1]); //th = 0
	
  delta[index + 0] =(-1.0) * scale * (tx - sigmoid(x[index + 0])) * sigmoid(x[index + 0]) * (1 - sigmoid(x[index + 0]));
  delta[index + 1] =(-1.0) * scale * (ty - sigmoid(x[index + 1])) * sigmoid(x[index + 1]) * (1 - sigmoid(x[index + 1]));
  delta[index + 2] =(-1.0) * scale * (tw - x[index + 2]);
  delta[index + 3] =(-1.0) * scale * (th - x[index + 3]);
  return iou;
}

template <typename Dtype>
void delta_yolov3region_class(Dtype* input_data, Dtype* &diff, int index, int class_label, int classes, float scale, Dtype* avg_cat, int focal_loss)
{
    Dtype tempsum = 0.0;
	int n = 0;
    if (diff[index])
    {
		diff[index + class_label] = 1 - input_data[index + class_label];
	    if (avg_cat)
	    {
			*avg_cat += input_data[index + class_label];
	    }
		return;
    }
	if (focal_loss)
	{
		float alpha = 0.5f;
		int ti = index + class_label;
		float pt = input_data[ti] + 0.000000000000001F;
		float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);
		for (n = 0; n < classes; n++)
		{
			diff[index + n] = ((n == class_label) ? 1 : 0) - input_data[index + n];
			diff[index + n] *= alpha*grad;
			if (n == class_label)
			{
				// avg_cat only for correct classification
				*avg_cat += input_data[index + n];
			}
		}
	}
	else
	{
		for (n = 0; n < classes; n++)
		{
			diff[index + n] = ((n == class_label) ? 1 : 0) - input_data[index + n];
			if (n == class_label && avg_cat) 
				*avg_cat += input_data[index + n];
		}
	}
}

template <typename Dtype>
void Yolov3LossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  Yolov3LossParameter param = this->layer_param_.yolov3_loss_param();
  
  side_ = bottom[0]->width();
  bias_match_ = param.bias_match(); //anchor boxes
  num_class_ = param.num_class(); //20
  coords_ = param.coords(); //4
  num_ = param.num(); //num of predicting boxes
  
  class_map_ = param.class_map();
  if (class_map_ != ""){
    string line;
    std::fstream fin(class_map_.c_str());
    if (!fin){
      LOG(INFO) << "no map file";
    }
    
    int index = 0;
    int id = 0;
    while (getline(fin, line)){
      stringstream ss;
      ss << line;
      ss >> id;
      
      cls_map_[index] = id;
      index ++;
    }
    fin.close();
  }  

  //LOG(INFO) << "t_.groups: " << t_.groups;
  //jitter_ = param.jitter(); 
  //rescore_ = param.rescore();
  
  object_scale_ = param.object_scale(); //5.0
  noobject_scale_ = param.noobject_scale(); //1.0
  class_scale_ = param.class_scale(); //1.0
  coord_scale_ = param.coord_scale(); //1.0
  
  //absolute_ = param.absolute();
  ignore_thresh_ = param.ignore_thresh(); //0.5
  truth_thresh_ = param.truth_thresh(); // 1.0
  focal_loss_ = param.focal_loss();
  max_boxes_ = param.max_boxes();
  random_ = param.random();  

  for (int c = 0; c < param.biases_size(); ++c) {
     biases_.push_back(param.biases(c)); 
  } //0.73 0.87;2.42 2.65;4.30 7.04;10.24 4.59;12.68 11.87;

  for (int m = 0; m < param.masks_size(); ++m) {
	  //masks are different at different yolo layers
	  masks_.push_back(param.masks(m));
  }

  int input_count = bottom[0]->count(1); //h*w*n*(classes+coords+1) = 13*13*5*(20+4+1), the size of the bottom input of region_loss_layer is h*w*filter_nums(filter_nums = 125)
  int label_count = bottom[1]->count(1); //30*5
  // outputs: classes, iou, coordinates
  int tmp_input_count = side_ * side_ * num_ * (coords_ + num_class_ + 1)*masks_.size()/num_; //13*13*5*(20+4+1) label: isobj, class_label, coordinates
  int tmp_label_count = max_boxes_ * (coords_+1);
  CHECK_EQ(input_count, tmp_input_count);//assertion checking
  CHECK_EQ(label_count, tmp_label_count);
}


template <typename Dtype>
void Yolov3LossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  LossLayer<Dtype>::Reshape(bottom, top);
  diff_.ReshapeLike(*bottom[0]);
  real_diff_.ReshapeLike(*bottom[0]); 
}

template <typename Dtype>
void Yolov3LossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* label_data = bottom[1]->cpu_data(); 
  Dtype* diff = diff_.mutable_cpu_data();
  caffe_set(diff_.count(), Dtype(0.0), diff);
  Dtype avg_noobj(0.0), avg_obj(0.0), avg_iou(0.0), avg_cat(0.0), recall(0.0), recall75(0.0), loss(0.0);
  int basiclength = bottom[0]->channels() / num_;
  int labelbaselength = 5;
  assert(basiclength == (5 + num_class_));
  int count = 0;
  int class_count = 0;
  Dtype fcount = 0.0;
  Dtype fclass_count = 0.0;
  Dtype best_iou = 0.0;
  int best_t = 0;
  int w_index = 0;
  int h_index = 0;
  //*********************************************************Reshape********************************************************//
  Blob<Dtype> swap;
  swap.Reshape(bottom[0]->num(), bottom[0]->height()*bottom[0]->width(), num_, bottom[0]->channels() / num_); 

  Dtype* swap_data = swap.mutable_cpu_data();
  int index = 0;
  for (int b = 0; b < bottom[0]->num(); ++b)
    for (int h = 0; h < bottom[0]->height(); ++h)
      for (int w = 0; w < bottom[0]->width(); ++w)
        for (int c = 0; c < bottom[0]->channels(); ++c)
        {
          swap_data[index++] = bottom[0]->data_at(b,c,h,w);	
        }
   
    
    //CHECK_EQ(bottom[0]->data_at(0,4,1,2),swap.data_at(0,15,0,4));
    //std::cout<<"5"<<std::endl;
    // yolov2在预测时会使用激活函数对结果进行激活，yolov3去掉了这些激活函数
    //disp(swap);
    //std::cout<<"7"<<std::endl;
    //disp(swap);
    //LOG(INFO) << "data ok!";
    //*********************************************************Diff********************************************************//
  int best_num = 0;
  for (int b = 0; b < swap.num(); ++b){
	  int true_index = b * bottom[1]->count(1);
	  vector<V3BoxData> gt_boxes;
	  GetGtFormLabelsimple(side_, labelbaselength, max_boxes_, label_data + true_index, &gt_boxes);
	  if (gt_boxes.size() > max_boxes_)
	  {
		  max_boxes_ = gt_boxes.size();
	  }
	  //先扫描一轮，计算出非目标网格中的结果
	  for (int h = 0; h < side_; h++)
	  {
		  for (int w = 0; w < side_; w++)
		  {
			  // indexed by num of box, channels = num_box*(4+1+classes)
			  for (int n = 0; n < num_; n++)
			  {
				  int image_index = (h*swap.width() + w);
				  int boxindex = b*swap.height()*swap.width()*swap.channels() + image_index*swap.channels() + n*basiclength + 0;
				  int obj_index = b*swap.height()*swap.width()*swap.channels() + image_index*swap.channels() + n*basiclength + 4;
				  int classindex = b*swap.height()*swap.width()*swap.channels() + image_index*swap.channels() + n*basiclength + 5;
				  vector<Dtype> pred = get_yolov3_box(swap_data, biases_, n, boxindex, h, w, side_, side_);
				  best_iou = 0.0;
				  best_t = 0;
				  for (int bb = 0; bb < max_boxes_; bb++)
				  {
					  vector<Dtype> truth;
					  Dtype x = gt_boxes[bb].box_[0];
					  Dtype y = gt_boxes[bb].box_[1];
					  Dtype w = gt_boxes[bb].box_[2];
					  Dtype h = gt_boxes[bb].box_[3];
					  if (!x) break;
					  truth.push_back(x);
					  truth.push_back(y);
					  truth.push_back(w);
					  truth.push_back(h);
					  Dtype iou = Calc_iou(pred, truth);
					  if (iou > best_iou)
					  {
						  best_iou = iou;
						  best_t = bb;
					  }
				  }
				  avg_noobj += swap_data[obj_index];
				  diff[obj_index] = 0 - swap_data[obj_index];
				  if (best_iou > ignore_thresh_)
				  {
					  diff[obj_index] = 0.0f;
				  }
				  if (best_iou > truth_thresh_)
				  {
					  //there should exist an object in the box, always, the program never enter here on the first round
					  diff[obj_index] = 1.0f - swap_data[obj_index];
					  int tclass_label = label_data[b * max_boxes_ * 5 + best_t * 5 + 1];
					  if (class_map_ != "") tclass_label = cls_map_[tclass_label];
					  delta_yolov3region_class(swap_data, diff, classindex, tclass_label, num_class_, class_scale_, &avg_cat, focal_loss_);
				  }
			  }
		  }
	  }
	  //第二轮计算有目标网格的结果
	  for (int t = 0; t < max_boxes_; t++)
	  {
		  vector<Dtype> truth_shift;
		  vector<Dtype> truth;
		  Dtype x = gt_boxes[t].box_[0];
		  Dtype y = gt_boxes[t].box_[1];
		  Dtype w = gt_boxes[t].box_[2];
		  Dtype h = gt_boxes[t].box_[3];
		  if (!x) break;
		  truth.push_back(x);
		  truth.push_back(y);
		  truth.push_back(w);
		  truth.push_back(h);
		  truth_shift.push_back(0.0);
		  truth_shift.push_back(0.0);
		  truth_shift.push_back(w);
		  truth_shift.push_back(h);
		  best_iou = 0.0;
		  best_t = 0;
		  w_index = (truth[0] * side_);//when you calc the loss of the real object, you only need to focus on the center coor of groudtruth
		  h_index = (truth[1] * side_);
		  for (int n = 0; n < num_; n++)
		  {
			  vector<Dtype> pred;
			  pred.push_back(0.0);
			  pred.push_back(0.0);
			  pred.push_back(biases_[2*n] / side_);
			  pred.push_back(biases_[2*n+1] / side_);
			  float iou = Calc_iou(pred, truth_shift);
			  if (iou > best_iou)
			  {
				  best_iou = iou;
				  best_t = n;
			  }
			  int mask_n = int_index(masks_, best_t, masks_.size());
			  if (mask_n >= 0)
			  {
				  int image_index = (h_index*swap.width() + w_index);
				  int box_index = b*swap.height()*swap.width()*swap.channels() + image_index*swap.channels() + mask_n*basiclength + 0;
				  vector<Dtype> pred = get_yolov3_box(swap_data, biases_, n, box_index, h_index, w_index, side_, side_);
				  float iou = Calc_iou(pred, truth);
				  int obj_index = box_index + 4;
				  avg_obj += swap_data[obj_index];
				  diff[obj_index] = 1 - swap_data[obj_index];
				  int tclass_label = label_data[b * max_boxes_ * 5 + n * 5 + 1];
				  if (class_map_ != "") tclass_label = cls_map_[tclass_label];
				  int class_index = box_index + 5;
				  delta_yolov3region_class(swap_data, diff, class_index, tclass_label, num_class_, class_scale_, &avg_cat, focal_loss_);

				  ++count;
				  ++class_count;
				  if (iou > .5) recall += 1;
				  if (iou > .75) recall75 += 1;
				  avg_iou += iou;
			  }
		  }
	  }
  }

  diff_.Reshape(bottom[0]->num(), bottom[0]->height()*bottom[0]->width(), num_, bottom[0]->channels() / num_);

  //使用real_diff计算backward
  Dtype* real_diff = real_diff_.mutable_cpu_data();
  int sindex = 0;

  //rerange
  for (int b = 0; b < real_diff_.num(); ++b)
	  for (int h = 0; h < real_diff_.height(); ++h)
		  for (int w = 0; w < real_diff_.width(); ++w)
			  for (int c = 0; c < real_diff_.channels(); ++c)
			  {
				  int rindex = b * real_diff_.height() * real_diff_.width() * real_diff_.channels() + c * real_diff_.height() * real_diff_.width() + h * real_diff_.width() + w;
				  Dtype e = diff[sindex];
				  real_diff[rindex] = e;
				  sindex++;
			  }

  for (int i = 0; i < real_diff_.count(); ++i)
  {
	  loss += real_diff[i] * real_diff[i];
  }
  top[0]->mutable_cpu_data()[0] = loss;
  iter_yolov3++;
  fcount = (Dtype)count;
  fclass_count = (Dtype)class_count;
  if (!(iter_yolov3 % 100))
  {
	  LOG(INFO) << "avg_noobj: " << avg_noobj / ((Dtype)(side_*side_*num_*bottom[0]->num())) << " avg_obj: " << avg_obj / fcount << " avg_iou: " << avg_iou / fcount << " avg_cat: " << avg_cat / fclass_count << " recall: " << recall / fcount << " recall75: " << recall75 / fcount << " class_count: " << class_count << " effective_count: " << count;
  }
}

template <typename Dtype>
void Yolov3LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //LOG(INFO) <<" propagate_down: "<< propagate_down[1] << " " << propagate_down[0];
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype sign(1.);
    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->num();
    //const Dtype alpha(1.0);
    //LOG(INFO) << "alpha:" << alpha;
    
    caffe_cpu_axpby(
        bottom[0]->count(),
        alpha,
        real_diff_.cpu_data(),
        Dtype(0),
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
//STUB_GPU(Yolov3LossLayer);
#endif

//INSTANTIATE_CLASS(Yolov3LossLayer);
REGISTER_LAYER_CLASS(Yolov3Loss);

}  // namespace caffe
