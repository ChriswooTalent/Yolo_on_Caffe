from __future__ import print_function, division

import argparse
import os
import math

voc_root = "D:/Code_local/caffe/caffe_yolo/lmdb/"
train_voc_list = voc_root + "trainval_lmdb"
test_voc_list = voc_root + "test2012_lmdb"

#coco_root =
#train_coco_list = my_project_root + "mnist/train/train.txt"
#test_coco_list = my_project_root + "mnist/test/test.txt"

if 'GLOG_minloglevel' not in os.environ:
    os.environ['GLOG_minloglevel'] = '2'  # suppress verbose Caffe output

from caffe import layers as cl
from caffe import layers as layer_test
from caffe import params as cp
import caffe

def load_configuration(fname):
    """ Load YOLO configuration file. """
    with open(fname, 'r') as fconf:
        lines = [l.strip() for l in fconf]

    config = []
    element = {}
    section_name = None
    for line in lines:
        if not line or line[0] == '#':  # empty or comment
            continue
        if line[0] == '[':  # new section
            if section_name:
                config.append((section_name, element))
                element = {}
            section_name = line[1:].strip(']')
        else:
            key, value = line.split('=')
            element[key] = value
    config.append((section_name, element))

    return config


## Layer parsing ##
###################

def data_label_layer(name, params, train=False):
    """ add a data-label layer """
    if train == False:
        prototag = caffe.TEST
    else:
        prototag = caffe.TRAIN
    if train:
        data_test = cl.BoxData(data_param=dict(batch_size=int(params["batch"]),
                                                              backend=cp.Data.LMDB,
                                                              source=train_voc_list),
                                              include=dict(phase=prototag,mirror=False),
                                              transform_param=dict(scale=1. / 255),
                                              ntop=2)
        return data_test
    else:
        fields = dict(shape={"dim": [1, int(params["channels"]),
                                     int(params["width"]), int(params["height"])]})
        data_test = cl.BoxData(data_param=dict(batch_size=int(params["batch"]),
                                                              mirror=False, backend=cp.Data.LMDB,
                                                              source=test_voc_list),
                                              include=dict(phase=prototag),
                                              transform_param=dict(scale=1. / 255),
                                              ntop=2)
        return data_test

def data_layer(name, params, train=False):
    """ add a data layer """
    fields = dict(shape={"dim": [1, int(params["channels"]),
                                 int(params["width"]), int(params["height"])]})
    return layer(name=name, **fields)


def activation_layer(previous, count, mode="relu"):
    """ create a non-linear activation layer """
    if   mode == "relu":
        return cl.RelU(previous, name="relu{}".format(count), in_place=True)
    elif mode == "leaky":
        return cl.ReLU(previous, name="relu{}".format(count),
                       in_place=True, relu_param=dict(negative_slope=0.1))
    else:
        raise ValueError("Activation mode not implemented: {0}".format(mode))


def convolutional_layer(previous, name, params, train=False, has_bn=False):
    """ create a convolutional layer given the parameters and previous layer """
    fields = dict(num_output=int(params["filters"]),
                  kernel_size=int(params["size"]))
    if "stride" in params.keys():
        fields["stride"] = int(params["stride"])
    if int(params.get("pad", 0)) == 1:    # use 'same' strategy for convolutions
        fields["pad"] = fields["kernel_size"]//2
    if has_bn:
        fields["bias_term"] = False

    if train:
        fields.update(weight_filler=dict(type="gaussian", std=0.01),
                      bias_filler=dict(type="constant", value=0))

    return cl.Convolution(previous, name=name, **fields)


def local_layer(previous, name, params, train=False):
    """ create a locally connected layer given the parameters and previous
    layer """
    if 'LocalConvolution' not in caffe.layer_type_list():
        raise ValueError("Layer not available: LocalConvolution")

    fields = dict(num_output=int(params["filters"]),
                  kernel_size=int(params["size"]))
    if "stride" in params.keys():
        fields["stride"] = int(params["stride"])

    if int(params.get("pad", 0)) == 1:    # use 'same' strategy for convolutions
        fields["pad"] = fields["kernel_size"]//2
    if train:
        fields.update(weight_filler=dict(type="gaussian", std=0.01),
                      bias_filler=dict(type="constant", value=0))

    return cl.LocalConvolution(previous, name=name, **fields)


def batchnorm_layer(previous, name, train=False):
    """ create a batch normalization layer given the parameters and previous
    layer """
    if not train:
        return cl.BatchNorm(previous, name=name, use_global_stats=True)

    return cl.BatchNorm(previous, name=name, include=dict(phase=caffe.TRAIN),
                        # suppress SGD on bn params for old Caffe versions
                        param=[dict(lr_mult=0, decay_mult=0)]*3,
                        use_global_stats=False)


def max_pooling_layer(previous, name, params):
    """ create a max pooling layer """
    return cl.Pooling(
        previous, name=name, pool=cp.Pooling.MAX,
        kernel_size=int(params["size"]), stride=int(params["stride"]))

def global_pooling_layer(previous, name, mode="avg"):
    """ create a Global Pooling Layer """
    pool = cp.Pooling.AVE if mode == "avg" else cp.Pooling.MAX
    return cl.Pooling(previous, name=name, pool=pool, global_pooling=True)

def dense_layer(previous, name, params, train=False):
    """ create a densse layer """
    fields = dict(num_output=int(params["output"]))
    if train:
        fields.update(weight_filler=dict(type="gaussian", std=0.01),
                      bias_filler=dict(type="constant", value=0))
    return cl.InnerProduct(previous, name=name, inner_product_param=fields)

def route_layer(previous, name, params, train=False):
    """create route layer of yolo"""
    comnon_layers = previous[0]
    conv_layers = previous[1]
    if "layers " in params.keys():
        layers_str = params["layers "]
        lstr_splited = layers_str.split(', ')
        layer_size = len(lstr_splited)
        list_layer = []
    if layer_size == 1:
        layer_index = int(lstr_splited[0])
        return cl.Concat(conv_layers[layer_index], name=name)
    elif layer_size == 2:
        layer_index1 = int(lstr_splited[0])
        layer_index2 = int(lstr_splited[1])
        return cl.Concat(conv_layers[layer_index1], conv_layers[layer_index2-1], name=name)

def reorg_layer(previous, name, params, train=False):
    """ create a reorg layer"""
    if "stride" in params.keys():
        fields = dict(stride = int(params["stride"]))
    return cl.Reorg(previous, name=name, stride=int(params["stride"]))

def shortcut_layer(previous, name, params, train=False):
    """create shortcut layer"""
    if "from" in params.keys():
        res_level = int(params["from"])
    comnon_layers = previous[0]
    conv_layers = previous[1]
    bottomlayers = (comnon_layers[-1], conv_layers[res_level])
    return cl.Eltwise(comnon_layers[-1], conv_layers[res_level], name=name, operation = 1)

def upsample_layer(previous, name, params, train=False):
    """ create a upsample layer given the parameters and previous layer """
    ustride = int(params["stride"])
    csize = 2*ustride-ustride%2
    cpad = math.ceil((ustride-1)/2.0)
    fields = dict(num_output=int(params["filters"]),
                  kernel_size=csize)
    fields["stride"] = ustride
    fields["pad"] = int(cpad)
    fields["bias_term"] = False

    if train:
        fields.update(weight_filler=dict(type="bilinear"),
                      bias_filler=dict(type="constant", value=0))

    return cl.Deconvolution(previous, name=name, **fields)

def region_loss_layer(previous, name, params, train=False):
    """ create region loss layer"""
    fields = dict(num_class=int(params["classes"]),
                  coords=int(params["coords"]))

    if "anchors " in params.keys():
        fields["biases"] = int(params["anchors "])

    if "jitter" in params.keys():
        fields["jitter"] = float(params["jitter"])

    if "num" in params.keys():
        fields["num"] = int(params["num"])

    if "object_scale" in params.keys():
        fields["object_scale"] = int(params["object_scale"])

    if "noobject_scale" in params.keys():
        fields["noobject_scale"] = int(params["noobject_scale"])

    if "class_scale" in params.keys():
        fields["class_scale"] = int(params["class_scale"])

    if "coord_scale" in params.keys():
        fields["coord_scale"] = int(params["coord_scale"])

    if "absolute" in params.keys():
        fields["absolute"] = int(params["absolute"])

    if "thresh" in params.keys():
        fields["thresh"] = float(params["thresh"])

    return cl.RegionLoss(previous, name=name, **fields)

def yolov3_loss_layer(previous, name, params, train=False):
    comnon_layers = previous[0]
    label_layers = previous[1]
    """ create region loss layer"""
    fields = dict(num_class=int(params["classes"]),
                  random=int(params["random"]),
                  num = int(params["num"]))

    if "anchors " in params.keys():
        anchor_size = len(params["anchors "])
        anchors_str = params["anchors "]
        str_splited = anchors_str.split(', ')
        anchor_size = len(str_splited)
        list_anchor = []
        for anchor_value in str_splited:
            avalue_splited = anchor_value.split(',')
            for av in avalue_splited:
                list_anchor.append(float(av))
        fields["biases"] = list_anchor

    if "mask " in params.keys():
        mask_size = len(params["mask "])
        mask_str = params["mask "]
        mstr_splited = mask_str.split(',')
        mask_size = len(mstr_splited)
        list_mask = []
        for mask_value in mstr_splited:
            list_mask.append(int(mask_value))
        fields["masks"] = list_mask

    #if "jitter" in params.keys():
    #   fields["jitter"] = float(params["jitter"])

    if "ignore_thresh " in params.keys():
        fields["ignore_thresh"] = float(params["ignore_thresh "])

    if "truth_thresh " in params.keys():
        fields["truth_thresh"] = float(params["truth_thresh "])

    return cl.Yolov3Loss(comnon_layers, label_layers, name=name, **fields)

def yolov3_detection_layer(previous, name, params, train=False):
    comnon_layers = previous[0]
    label_layers = previous[1]
    fields = dict(num_class=int(params["classes"]),
                  num=int(params["num"]))

    if "anchors " in params.keys():
        anchor_size = len(params["anchors "])
        anchors_str = params["anchors "]
        str_splited = anchors_str.split(', ')
        anchor_size = len(str_splited)
        list_anchor = []
        for anchor_value in str_splited:
            avalue_splited = anchor_value.split(',')
            for av in avalue_splited:
                list_anchor.append(float(av))
        fields["biases"] = list_anchor

    if "mask " in params.keys():
        mask_size = len(params["mask "])
        mask_str = params["mask "]
        mstr_splited = mask_str.split(',')
        mask_size = len(mstr_splited)
        list_mask = []
        for mask_value in mstr_splited:
            list_mask.append(int(mask_value))
        fields["masks"] = list_mask

    return cl.Yolov3Detect(comnon_layers, label_layers, name=name, **fields)

### layer aggregation ###
#########################
def add_yolov3loss_layer(layers, bottom_inputs, count, params, train=False):
    """add layers related to yolov3 loss block in YOLO the layer list"""
    layer_name = "yolo{0}".format(count)
    layers.append(yolov3_loss_layer(bottom_inputs, layer_name, params, train))

def add_regionloss_layer(layers, count, params, train=False):
    """add layers related to a region block in YOLO the layer list"""
    layer_name = "region{0}".format(count)
    layers.append(region_loss_layer(layers[-1], layer_name, params, train))

def add_reorg_layer(layers, count, params, train=False):
    """add layers related to a reorg block in YOLO the layer list"""
    layer_name = "reorg{0}".format(count)
    layers.append(reorg_layer(layers[-1], layer_name, params, train))

def add_convolutional_layer(layers, count, params, train=False):
    """ add layers related to a convolutional block in YOLO the layer list """
    layer_name = "conv{0}".format(count)
    has_batch_norm = (params.get("batch_normalize", '0') == '1')

    layers.append(convolutional_layer(layers[-1], layer_name, params,
                                      train, has_batch_norm))
    if has_batch_norm:
        layers.append(batchnorm_layer(layers[-1], "{0}_bn".format(layer_name),
                                      train))
        layers.append(cl.Scale(layers[-1], name="{0}_scale".format(layer_name),
                               scale_param=dict(bias_term=True)))
    if params["activation"] != "linear":
        layers.append(activation_layer(layers[-1], count, params["activation"]))


def add_dense_layer(layers, count, params, train=False):
    """ add layers related to a connected block in YOLO to the layer list """
    layers.append(dense_layer(layers[-1], "fc{0}".format(count), params, train))
    if params["activation"] != "linear":
        layers.append(activation_layer(layers[-1], count, params["activation"]))


def add_local_layer(layers, count, params, train=False):
    """ add layers related to a connected block in YOLO to the layer list """
    layers.append(local_layer(layers[-1], "local{0}".format(count), params, train))
    if params["activation"] != "linear":
        layers.append(activation_layer(layers[-1], count, params["activation"]))

def add_route_layer(layers, bottom_inputs, count, params, train):
    """add route layers of yolo to implement the sum of two layers"""
    layer_name = "route{0}".format(count)
    layers.append(route_layer(bottom_inputs, layer_name, params, train))

def add_shortcut_layer(layers, bottom_inputs, count, params, train):
    """ add shortcut layers of yolov3 to implement the residual net"""
    layer_name = "res{0}".format(count)
    layers.append(shortcut_layer(bottom_inputs, layer_name, params, train))

def add_upsample_layer(layers, count, params, train=False):
    """add upsample layers of yolov3 to the layer list"""
    layer_name = "upsample{0}".format(count)
    layers.append(upsample_layer(layers[-1], layer_name, params, train))

def add_yolodetection_layer(layers, bottom_inputs, count, params, train):
    layer_name = "yolodetection{0}".format(count)
    layers.append(yolov3_detection_layer(bottom_inputs, layer_name, params, train))


def convert_configuration(config, train=False, single_deploy = False, loc_layer=False):
    """ given a list of YOLO layers as dictionaries, convert them to Caffe """
    layers = []
    conv_layers = []
    total_layers = []
    count = 0
    reorg_count = 0
    concat_count = 0
    shortcut_count = 0
    upsample_count = 0
    yolo_count = 0
    region_count = 0
    conv_count = 0
    last_conv_filters = 0;
    model = caffe.NetSpec()

    for section, params in config:
        if   section == "net":
            input_params = params
            if single_deploy == False:
                layertrained = data_label_layer("data", input_params, train)
                model.data = layertrained[0]
                model.labels = layertrained[1]
                layers.append(model.data)
            else:
                layers.append(data_layer("data", input_params, train))
        elif section == "crop":
            if train:    # update data layer with crop parameters
                input_params.update(params)
                #layers[-1] = data_layer("data", input_params, train)
                layers[-2:-1] = data_label_layer("data", input_params, train)
        elif section == "convolutional":
            conv_count += 1
            count += 1
            last_conv_filters = int(params["filters"])
            add_convolutional_layer(layers, count, params, train)
            conv_layers.append(layers[-1])
            total_layers.append(layers[-1])
        elif section == "maxpool":
            layers.append(max_pooling_layer(layers[-1], "pool{0}".format(count),
                                            params))
            total_layers.append(layers[-1])
        elif section == "avgpool":
            layers.append(global_pooling_layer(layers[-1], "pool{0}".format(count)))
            total_layers.append(layers[-1])
        elif section == "softmax":
            layers.append(cl.Softmax(layers[-1], name="softmax{0}".format(count)))
            total_layers.append(layers[-1])
        elif section == "connected":
            count += 1
            add_dense_layer(layers, count, params, train)
            total_layers.append(layers[-1])
        elif section == "route":
            concat_count += 1
            bottom_inputs = [layers, total_layers]
            add_route_layer(layers, bottom_inputs, concat_count, params, train)
            total_layers.append(layers[-1])
        elif section == 'shortcut':
            shortcut_count += 1
            bottom_inputs = [layers, total_layers]
            add_shortcut_layer(layers, bottom_inputs, shortcut_count, params, train)
            total_layers.append(layers[-1])
        elif section == 'upsample':
            upsample_count += 1
            dictvalue = {"filters", last_conv_filters}
            params.update({"filters":last_conv_filters})
            add_upsample_layer(layers, upsample_count, params, train)
            total_layers.append(layers[-1])
        elif section == 'yolo':
            yolo_count += 1
            bottom_inputs = [layers[-1], model.labels]
            if train == False:
                add_yolodetection_layer(layers, bottom_inputs, yolo_count, params, train)
            else:
                add_yolov3loss_layer(layers, bottom_inputs, yolo_count, params, train)
            total_layers.append(layers[-1])
        elif section == "reorg":
            reorg_count += 1
            add_reorg_layer(layers, reorg_count, params, train)
            total_layers.append(layers[-1])
        elif section == "region":
            region_count += 1
            add_regionloss_layer(layers, reorg_count, params, train)
            total_layers.append(layers[-1])
        elif section == "dropout":
            if train:
                layers.append(cl.Dropout(layers[-1], name="drop{0}".format(count),
                                         dropout_ratio=float(params["probability"])))
                total_layers.append(layers[-1])
        elif section == "local" and loc_layer:
            count += 1
            add_local_layer(layers, count, params, train)
            total_layers.append(layers[-1])
        else:
            print("WARNING: {0} layer not recognized".format(section))

    debug_count = 0
    if single_deploy == False:
        weight_layers = layers[1:]
    else:
        weight_layers = layers
    for layer in weight_layers:
        debug_count = debug_count+1
        print(debug_count)
        if debug_count == 323:
            print("fuck you!")
        setattr(model, layer.fn.params["name"], layer)
    model.result = layers[-1]

    return model


def adjust_params(model, model_filename):
    """ Set layer parameters that depends on blob attributes.
    Blobs are available only in Net() objects, but NetSpec() or NetParameters()
    can't be used to create a Net(). So we write a first prototxt, we reload it,
    fix the missing parameters and save it again.
    """
    with open(model_filename, 'w') as fproto:
        fproto.write("{0}".format(model.to_proto()))

    net = caffe.Net(model_filename, caffe.TEST)
    for name, layer in model.tops.iteritems():
        if name.startswith("local"):
            width, height = net.blobs[name].data.shape[-2:]
            if width != height:
                raise ValueError(" Only square inputs supported for local layers.")
            layer.fn.params.update(
                local_region_number=width, local_region_ratio=1.0/width,
                local_region_step=1)

    return model


def main():
    """ script entry point """
    parser = argparse.ArgumentParser(description='Convert a YOLO cfg file.')
    parser.add_argument('model', type=str, help='YOLO cfg model')
    parser.add_argument('output', type=str, help='output prototxt')
    parser.add_argument('--loclayer', action='store_true',
                        help='use locally connected layer')
    parser.add_argument('--train', action='store_true',
                        help='generate train_val prototxt')
    args = parser.parse_args()

    config = load_configuration(args.model)
    caffe.set_mode_cpu()
    #layers_test = []
    #cl.Convolution(layers_test[-1], kernel_size=3)
    layer_type = caffe.layer_type_list()
    for temp in layer_type:
        if temp == "Reorg":
            print("{0} layer recognized".format(temp))
        elif temp == "convolutional":
            print("{0} layer recognized".format(temp))
    args.train = True
    deploy = False
    model = convert_configuration(config, args.train, deploy, args.loclayer)

    suffix = "train_val" if args.train else "deploy"
    model_filename = "{}_{}.prototxt".format(args.output, suffix)

    if args.loclayer:
        model = adjust_params(model, model_filename)

    with open(model_filename, 'w') as fproto:
        fproto.write("{0}".format(model.to_proto()))


if __name__ == '__main__':
    main()