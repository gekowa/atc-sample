/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "register/register.h"

#define OP_LOGE(OP_NAME, fmt, ...) printf("[ERROR]%s,%s:%u:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define OP_LOGW(OP_NAME, fmt, ...) printf("[WARN]%s,%s:%u:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define OP_LOGI(OP_NAME, fmt, ...) printf("[INFO]%s,%s:%u:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)

namespace domi {
    namespace {
        const char *const kBoxesUnpack = "/unstack";
        const char *const kBoxesDiv = "RealDiv";
        const size_t kRealDivInputSize = 2;
        const size_t kScaleSize = 4;
    }  // namespace

    Status ParseFloatFromConstNode(const ge::Operator *node, float &value) {
        if (node == nullptr) {
            return FAILED;
        }
        ge::Tensor tensor;
        auto ret = node->GetAttr("value", tensor);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE(node->GetName().c_str(), "Failed to get value from %s", node->GetName().c_str());
            return FAILED;
        }
        uint8_t *data_addr = tensor.GetData();
        value = *(reinterpret_cast<float *>(data_addr));
        return SUCCESS;
    }

    Status DecodeBboxV2ParseParams(const std::vector <ge::Operator> &inside_nodes, ge::Operator &op_dest) {
        std::map <std::string, std::string> scales_const_name_map;
        std::map<string, const ge::Operator *> node_map;
        for (const auto &node : inside_nodes) {
            if (node.GetOpType() == kBoxesDiv) {
                if (node.GetInputsSize() < kRealDivInputSize) {
                    OP_LOGE(node.GetName().c_str(), "Input size of %s is invalid, which is %zu.", kBoxesDiv,
                            node.GetInputsSize());
                    return FAILED;
                }
                auto input_unpack_name = node.GetInputDesc(0).GetName();
                if (input_unpack_name.find(kBoxesUnpack) != string::npos) {
                    scales_const_name_map.insert({node.GetName(), node.GetInputDesc(1).GetName()});
                }
            }
            node_map[node.GetName()] = &node;
        }

        std::vector<float> scales_list = {1.0, 1.0, 1.0, 1.0};
        if (scales_const_name_map.size() != kScaleSize) {
            OP_LOGI(op_dest.GetName().c_str(), "Boxes doesn't need scale.");
        } else {
            size_t i = 0;
            for (const auto &name_pair : scales_const_name_map) {
                float scale_value = 1.0;
                auto ret = ParseFloatFromConstNode(node_map[name_pair.second], scale_value);
                if (ret != SUCCESS) {
                    return ret;
                }
                scales_list[i++] = scale_value;
            }
        }
        op_dest.SetAttr("scales", scales_list);
        return SUCCESS;
    }

    REGISTER_CUSTOM_OP("DecodeBboxV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodeBboxV2FusionOp")
    .FusionParseParamsFn(DecodeBboxV2ParseParams)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
