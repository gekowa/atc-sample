/**
 * Copyright 2020 Huawei Technologies Co., Ltd
*/

#include "register/register.h"
#include "graph/operator.h"
#include <string>

using namespace ge;
namespace domi {

const std::string NUM_OUTPUT = "num_output";
const std::string GROUP = "group";
const std::string KERNEL_SIZE = "kernel_size";
const std::string KERNEL_H = "kernel_h";
const std::string KERNEL_W = "kernel_w";
const std::string AXiS = "axis";
const std::string FORCE_ND_IM2COL = "force_nd_im2col";
const std::string PAD = "pad";
const std::string PAD_H = "pad_h";
const std::string PAD_W = "pad_w";
const std::string PADS = "pads";
const std::string STRIDE = "stride";
const std::string STRIDE_H = "stride_h";
const std::string STRIDE_W = "stride_w";
const std::string STRIDES = "strides";
const std::string DILATION = "dilation";
const std::string DILATIONS = "dilations";
// if op_src get required attr failed, need to return Failed
// if op_src get optional attr failed, need to return Failed or set a default value
// Get covolution pad params from caffe proto and convert to tbe conv2d ir
// pad flag [pads]
static bool SetPads(const ge::Operator& op_src, ge::Operator& op_dest)
{
    const int kDefaultPad = 0;
    int64_t pad[2] = {kDefaultPad, kDefaultPad};
    std::vector<int64_t> pad_attr;
    int pad_h;
    int pad_w;
    if (ge::GRAPH_SUCCESS != op_src.GetAttr(PAD, pad_attr)){
        return false;
    }
    const int pSize = pad_attr.size();
    if (ge::GRAPH_SUCCESS == op_src.GetAttr(PAD_H, pad_h) || ge::GRAPH_SUCCESS == op_src.GetAttr(PAD_W, pad_w)){
        if (pSize != 0) {
            return false;
        }
        pad[0] = pad_h;
        pad[1] = pad_w;
    }else{

        if (pSize == 1 || pSize == 2) {
            for (size_t i = 0; i < 2; i++) {
                int index = (pSize == 1) ? 0 : i;
                pad[i] = pad_attr[index];
            }
        } else if (pSize != 0) {

            return false;
        }
    }

    std::vector<int64_t> pList;
    pList.push_back(pad[0]);
    pList.push_back(pad[0]);
    pList.push_back(pad[1]);
    pList.push_back(pad[1]);
    op_dest.SetAttr(PADS, (pList));
    return true;
}

// Get covolution stride params from caffe proto and convert to tbe conv2d
// ir [strides]
static bool SetStrides(const ge::Operator& op_src, ge::Operator& op_dest)
{

    const int kDefaultStride = 1;
    int64_t stride[2] = {kDefaultStride, kDefaultStride};
    std::vector<int64_t> stride_attr;
    if (ge::GRAPH_SUCCESS != op_src.GetAttr(STRIDE, stride_attr)){
        return false;
    }

    const int sSize= stride_attr.size();
    int stride_h;
    int stride_w;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr(STRIDE_H, stride_h) || ge::GRAPH_SUCCESS == op_src.GetAttr(STRIDE_W, stride_w)){
        if (sSize != 0) {
            return false;
        }
        stride[0] = stride_h;
        stride[1] = stride_w;
    }else {
        if (sSize == 1 || sSize == 2) {
            for (size_t i = 0; i < 2; i++) {
                int index = (sSize == 1) ? 0 : i;
                stride[i] = stride_attr[index];
            }
        } else if (sSize != 0) {
            return false;
        }
    }
    std::vector<int64_t> sList;
    sList.push_back(1);
    sList.push_back(1);
    sList.push_back(stride[0]);
    sList.push_back(stride[1]);
    op_dest.SetAttr(STRIDES, (sList));

    return true;
}

// Get covolution dilation params from caffe proto and convert to tbe conv2d
// ir [dilations]
static bool SetDilations(const ge::Operator& op_src, ge::Operator& op_dest)
{
    const int kDefaultDilation = 1;
    std::vector<int64_t> dilation_attr;
    int64_t dilation[2] = {kDefaultDilation, kDefaultDilation};
    if (ge::GRAPH_SUCCESS == op_src.GetAttr(DILATION, dilation_attr)){
        const int dSize = dilation_attr.size();
        if (dSize == 1 || dSize == 2) {
            for (size_t i = 0; i < 2; i++) {
                int index = (dSize == 1) ? 0 : i;
                dilation[i] = dilation_attr[index];
            }
        } else if (dSize != 0) {
            return false;
        }
    }

    std::vector<int64_t> dList;
    dList.push_back(1);
    dList.push_back(1);
    dList.push_back(dilation[0]);
    dList.push_back(dilation[1]);
    op_dest.SetAttr(DILATIONS, (dList));

    return true;

}

// Check input parameters that are illegal or not applicable to 2D convolution
static bool ProcSpecParams(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int num_output;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr(NUM_OUTPUT, num_output)){
        if (num_output < 1) {
            return false;
        }
    }
    int group;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr(GROUP, group)){
        if (group < 1 || num_output % group != 0) {
            return false;
        }
        op_dest.SetAttr(GROUP, (int64_t)group);
    }


    vector<int64_t> kernel_size;
    if (ge::GRAPH_SUCCESS != op_src.GetAttr(KERNEL_SIZE, kernel_size)){
        return false;
    }
    int kSize = kernel_size.size();

    int kernel[2] = {0, 0};
    int kernel_h;
    int kernel_w;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr(KERNEL_H, kernel_h) || ge::GRAPH_SUCCESS == op_src.GetAttr(KERNEL_W, kernel_w)){
        if (kSize != 0) {
            return false;
        }
        kernel[0] = kernel_h;
        kernel[1] = kernel_w;
    }else{

        if (kSize == 1 || kSize == 2) {
            for (size_t i = 0; i < 2; i++) {
                int index = (kSize == 1) ? 0 : i;
                kernel[i] = kernel_size[index];
            }
        } else {
            return false;
        }
    }

    for (size_t i = 0; i < 2; i++) {
        if (kernel[i] < 1) {
            return false;
        }
    }

    int channel_axis;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr(AXiS, channel_axis)){
        if ((channel_axis + 4) % 4 != 1) {
            return false;
       }
    }
    bool force_nd_im2col;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr(FORCE_ND_IM2COL, force_nd_im2col)){
        if (force_nd_im2col) {
            return false;
        }
    }
    return true;
}

// Replace GE ParseParams fuction to process graph conv2d node attrs
Status ParseParamsConv2D(const ge::Operator& op_src, ge::Operator& op_dest)
{

    if (!(ProcSpecParams(op_src, op_dest) && SetPads(op_src, op_dest) &&
          SetStrides(op_src, op_dest) && SetDilations(op_src, op_dest))) {
        return FAILED;
    }

    return SUCCESS;
}

REGISTER_CUSTOM_OP("Conv2DTik")
    .FrameworkType(CAFFE)  // type: CAFFE, TENSORFLOW
    .OriginOpType("ConvolutionTik")  // name in caffe module
    .ParseParamsByOperatorFn(ParseParamsConv2D)  // AutoMappingFn for Tensorflow,
    // ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}  // namespace domi
