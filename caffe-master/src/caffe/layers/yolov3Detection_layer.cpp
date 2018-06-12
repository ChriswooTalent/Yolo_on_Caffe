#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layers/detection_loss_layer.hpp"
#include "caffe/layers/yolov3Detection_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {
	bool BoxSortDecendScore(const V3BoxData& box1, const V3BoxData& box2) {
		return box1.score_ > box2.score_;
	}

	void ApplyNms(const vector<V3BoxData>& boxes, vector<int>* idxes, float threshold) {
		map<int, int> idx_map;
		for (int i = 0; i < boxes.size() - 1; ++i) {
			if (idx_map.find(i) != idx_map.end()) {
				continue;
			}
			vector<float> box1 = boxes[i].box_;
			for (int j = i + 1; j < boxes.size(); ++j) {
				if (idx_map.find(j) != idx_map.end()) {
					continue;
				}
				vector<float> box2 = boxes[j].box_;
				float iou = Calc_iou(box1, box2);
				if (iou >= threshold) {
					idx_map[j] = 1;
				}
			}
		}
		for (int i = 0; i < boxes.size(); ++i) {
			if (idx_map.find(i) == idx_map.end()) {
				idxes->push_back(i);
			}
		}
	}

	template<typename Dtype>
	void Correct_yolo_box(vector<Dtype> box, int w, int h, int netw, int newh, int relative, int letter)
	{

	}

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
				gt_box.score_ = (float)gridindex/locations;
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
	void yolov3DetectionLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		Yolov3DetectParameter param = this->layer_param_.yolov3_detection_param();
		side_ = param.side();
		num_class_ = param.num_class();
		num_object_ = param.num();
		coords_ = param.coords();
		obj_threshold_ = param.obj_thresh();
		nms_threshold_ = param.nms_thresh();
		nms_ = param.nms();

		max_boxes_ = param.max_boxes();

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
		int tmp_input_count = side_ * side_ * num_object_ * (coords_ + num_class_ + 1); //13*13*5*(20+4+1) label: isobj, class_label, coordinates
		int tmp_label_count = max_boxes_ * num_object_;
		CHECK_EQ(input_count, tmp_input_count);//assertion checking
		CHECK_EQ(label_count, tmp_label_count);
	}

	template <typename Dtype>
	void yolov3DetectionLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		int input_count = bottom[0]->count(1);
		int label_count = bottom[1]->count(1);
		// outputs: classes, iou, coordinates
		int tmp_input_count = side_ * side_ * num_object_* (coords_ + num_class_ + 1);
		// label: isobj, class_label, coordinates
		int tmp_label_count = max_boxes_ * num_object_;
		CHECK_EQ(input_count, tmp_input_count);
		CHECK_EQ(label_count, tmp_label_count);

		vector<int> top_shape(2, 1);
		top_shape[0] = bottom[0]->num();
		top_shape[1] = side_ * side_ * (coords_ + 4);
		top[0]->Reshape(top_shape);

		temp_top_.ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void yolov3DetectionLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const Dtype* input_data = bottom[0]->cpu_data();
		const Dtype* label_data = bottom[1]->cpu_data();
		//*********************************************************Reshape********************************************************//
		Blob<Dtype> temp_bottom;
		temp_bottom.Reshape(bottom[0]->num(), bottom[0]->height()*bottom[0]->width(), num_object_, bottom[0]->channels() / num_object_);

		int basiclength = bottom[0]->channels() / num_object_;
		int label_basiclength = 5;

		Dtype *temp_bottomdata = temp_bottom.mutable_cpu_data();
		int temp_index = 0;
		for (int n = 0; n < bottom[0]->num(); n++)
		{
			for (int i = 0; i < bottom[0]->height(); i++)
			{
				for (int j = 0; j < bottom[0]->width(); j++)
				{
					for (int c = 0; c < bottom[0]->channels(); c++)
					{
						temp_bottomdata[temp_index++] = bottom[0]->data_at(n, c, i, j);
					}
				}
			}
		}
		Dtype* top_data = top[0]->mutable_cpu_data();
		caffe_set(top[0]->count(), Dtype(0), top_data);
		for (int i = 0; i < bottom[0]->num(); ++i) {
			int input_index = i * bottom[0]->count(1);
			int true_index = i * bottom[1]->count(1);
			int top_index = i * top[0]->count(1);
			map<int, vector<V3BoxData> > gt_boxes;
			GetYolov3GTBox(side_, label_basiclength, label_data + true_index, &gt_boxes);
			for (std::map<int, vector<V3BoxData > >::iterator it = gt_boxes.begin(); it != gt_boxes.end(); ++it) {
				int label = it->first;
				vector<V3BoxData>& g_boxes = it->second;
				for (int j = 0; j < g_boxes.size(); ++j) {
					top_data[top_index + label] += 1;
				}
			}
			map<int, vector<V3BoxData> > pred_boxes;
			//this step has finished nms already
			GetYolov3PredBoxes(side_, num_object_, num_class_, basiclength, temp_bottomdata + input_index, &pred_boxes, biases_, score_type_, nms_, obj_threshold_, nms_threshold_);
			int index = top_index + num_class_;
			int pred_count(0);
			for (std::map<int, vector<V3BoxData> >::iterator it = pred_boxes.begin(); it != pred_boxes.end(); ++it) {
				int label = it->first;
				vector<V3BoxData>& p_boxes = it->second;
				//如果预测的label不在groundtruth中，增加预测结果的个数
				if (gt_boxes.find(label) == gt_boxes.end()) {
					for (int b = 0; b < p_boxes.size(); ++b) {
						top_data[index + pred_count * 4 + 0] = p_boxes[b].label_;
						top_data[index + pred_count * 4 + 1] = p_boxes[b].score_;
						top_data[index + pred_count * 4 + 2] = 0;
						top_data[index + pred_count * 4 + 3] = 1;
						top_data[index + pred_count * 4 + 4] = p_boxes[b].box_[0];
						top_data[index + pred_count * 4 + 5] = p_boxes[b].box_[1];
						top_data[index + pred_count * 4 + 6] = p_boxes[b].box_[2];
						top_data[index + pred_count * 4 + 7] = p_boxes[b].box_[3];
						++pred_count;
					}
					continue;
				}
				//正常预测结果的统计
				vector<V3BoxData>& g_boxes = gt_boxes[label];
				vector<bool> records(g_boxes.size(), false);
				for (int k = 0; k < p_boxes.size(); ++k) {
					top_data[index + pred_count * 4 + 0] = p_boxes[k].label_;
					top_data[index + pred_count * 4 + 1] = p_boxes[k].score_;
					top_data[index + pred_count * 4 + 4] = p_boxes[k].box_[0];
					top_data[index + pred_count * 4 + 5] = p_boxes[k].box_[1];
					top_data[index + pred_count * 4 + 6] = p_boxes[k].box_[2];
					top_data[index + pred_count * 4 + 7] = p_boxes[k].box_[3];
					float max_iou(-1);
					int idx(-1);
					for (int g = 0; g < g_boxes.size(); ++g) {
						float iou = Calc_iou(p_boxes[k].box_, g_boxes[g].box_);
						if (iou > max_iou) {
							max_iou = iou;
							idx = g;
						}
					}
					if (max_iou >= obj_threshold_) {
						if (!records[idx]) {
							records[idx] = true;
							top_data[index + pred_count * 4 + 2] = 1;
							top_data[index + pred_count * 4 + 3] = 0;
						}
						else {
							top_data[index + pred_count * 4 + 2] = 0;
							top_data[index + pred_count * 4 + 3] = 1;
						}
					}
					else {
						top_data[index + pred_count * 4 + 2] = 0;
						top_data[index + pred_count * 4 + 3] = 1;
					}
					++pred_count;
				}
			}
		}
	}

	INSTANTIATE_CLASS(yolov3DetectionLayer);
	REGISTER_LAYER_CLASS(yolov3Detection);

}  // namespace caffe
