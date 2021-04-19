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

#include "decode_bbox_v2_multi_pass.h"

#define OP_LOGE(OP_NAME, fmt, ...) printf("[ERROR]%s,%s:%u:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define OP_LOGW(OP_NAME, fmt, ...) printf("[WARN]%s,%s:%u:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define OP_LOGI(OP_NAME, fmt, ...) printf("[INFO]%s,%s:%u:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)

namespace ge {
    namespace {
        const char *const kScopeType = "DecodeBboxV2";
        const char *const kScopeTypeDecodeBboxV2 = "DecodeBboxV2";
        const char *const kOpType = "DecodeBboxV2";
        const char *const kBoxesUnpack = "/unstack";
        const char *const kBoxesDiv = "RealDiv";
        const size_t kRealDivInputSize = 2;
        const size_t kScaleSize = 4;
    }  // namespace

    std::vector<ScopeFusionPatterns> DecodeBboxV2MultiScopeFusionPass::DefinePatterns() {
        std::vector<ScopeFusionPatterns> patterns_list;
        ScopeFusionPatterns pattern;
        GenScopePatterns(pattern);
        patterns_list.push_back(pattern);
        return patterns_list;
    }


    void DecodeBboxV2MultiScopeFusionPass::GenScopePatterns(ScopeFusionPatterns &patterns) {
        std::vector<ScopePattern *> batch;
        ScopePattern *decode_bbox_v2_pattern = new(std::nothrow) ScopePattern();
        if (decode_bbox_v2_pattern == nullptr) {
            OP_LOGE(kOpType, "Alloc an object failed.");
            return;
        }
        decode_bbox_v2_pattern->SetSubType(kScopeTypeDecodeBboxV2);
        decode_bbox_v2_pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Exp", 2, 0));        // Exp num is 2
        decode_bbox_v2_pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Mul", 4, 0));        // Mul num is 4
        decode_bbox_v2_pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Sub", 4, 0));        // Sub num is 4
        decode_bbox_v2_pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("RealDiv", 0, 2));    // RealDiv num is 2*n
        decode_bbox_v2_pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Unpack", 2, 0));     // Unpack num is 2
        decode_bbox_v2_pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Pack", 1, 0));       // Pack num is 1
        decode_bbox_v2_pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Transpose", 3, 0));  // Transpose num is 3
        decode_bbox_v2_pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Softmax", -1, 0));   // doesn't have Softmax

        OP_LOGI(kOpType, "Add GenScopePatterns DecodeBboxV2.");
        batch.push_back(decode_bbox_v2_pattern);
        patterns.push_back(batch);
    }

    std::string DecodeBboxV2MultiScopeFusionPass::PassName() { return std::string("DecodeBboxV2MultiScopeFusionPass"); }

    Status DecodeBboxV2MultiScopeFusionPass::LastMatchScopesAndOPs(shared_ptr<ScopeGraph> &scope_graph,
                                                              std::vector<ScopesResult> &results) {
        OP_LOGI(kOpType, "LastMatchScopesAndOPs start.");
        if (scope_graph == nullptr) {
            OP_LOGE(kOpType, "Input params is nullptr.");
            return FAILED;
        }
        const ScopeTree *scope_tree = scope_graph->GetScopeTree();
        if (scope_tree == nullptr) {
            OP_LOGE(kOpType, "Scope tree is nullptr.");
            return FAILED;
        }
        const std::vector<Scope *> &scopes = scope_tree->GetAllScopes();

        for (auto &scope : scopes) {
            // Class ScopeTree guarantees scope is not empty.
            if (scope->SubType() == kScopeTypeDecodeBboxV2) {
                OP_LOGI(kOpType, "DecodeBbox LastMatchScopesAndOPs match scope %s.", scope->Name().c_str());
                ScopesResult result;
                std::vector<Scope *> result_scopes;
                result_scopes.push_back(scope);
                result.SetScopes(result_scopes);
                std::vector<ge::OperatorPtr> nodes;
                for (const auto &node_info : scope->AllNodesMap()) {
                    nodes.emplace_back(node_info.second);
                }
                result.SetNodes(nodes);
                results.push_back(result);
            }
        }
        return (!(results.empty())) ? SUCCESS : FAILED;
    }

    namespace {
        Status ParseFloatFromConstNode(const ge::OperatorPtr node, float &value) {
            if (node == nullptr) {
                return FAILED;
            }
            ge::Tensor tensor;
            auto ret = node->GetAttr("value", tensor);
            if (ret != ge::GRAPH_SUCCESS) {
                OP_LOGE(kOpType, "Failed to get value from %s", node->GetName().c_str());
                return FAILED;
            }
            uint8_t *data_addr = tensor.GetData();
            value = *(reinterpret_cast<float *>(data_addr));
            return SUCCESS;
        }

        Status DecodeBboxV2ParseParams(const std::vector<ge::OperatorPtr> &inside_nodes, ge::Operator *op_dest) {
            if (op_dest == nullptr) {
                OP_LOGE(kOpType, "Dest operator is nullptr.");
                return FAILED;
            }
            std::map<std::string, std::string> scales_const_name_map;
            std::map<string, ge::OperatorPtr> node_map;
            for (const auto &node : inside_nodes) {
                if (node == nullptr) {
                    OP_LOGE(kOpType, "Inner operator is nullptr.");
                    return FAILED;
                }
                if (node->GetOpType() == kBoxesDiv) {
                    if (node->GetInputsSize() < kRealDivInputSize) {
                        OP_LOGE(kOpType, "Input size of %s is invalid, which is %zu.", kBoxesDiv,
                                node->GetInputsSize());
                        return FAILED;
                    }
                    auto input_unpack_name = node->GetInputDesc(0).GetName();
                    if (input_unpack_name.find(kBoxesUnpack) != string::npos) {
                        scales_const_name_map.insert({node->GetName(), node->GetInputDesc(1).GetName()});
                    }
                }
                node_map[node->GetName()] = node;
            }

            std::vector<float> scales_list = {1.0, 1.0, 1.0, 1.0};
            if (scales_const_name_map.size() != kScaleSize) {
                OP_LOGI(op_dest->GetName().c_str(), "Boxes doesn't need scale.");
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
            op_dest->SetAttr("scales", scales_list);
            return SUCCESS;
        }
    }  // namespace

    void DecodeBboxV2MultiScopeFusionPass::GenerateFusionResult(const std::vector<Scope *> &scopes,
                                                           FusionScopesResult *fusion_rlt) {
        if (fusion_rlt == nullptr) {
            return;
        }
        if (scopes.size() != 1) {
            // not match, set
            fusion_rlt->SetType(kScopeInvalidType);
            return;
        }


        fusion_rlt->InsertInputs("transpose", {0, kFusionDisableIndex});
        fusion_rlt->InsertInputs("get_center_coordinates_and_sizes/transpose", {1, kFusionDisableIndex});
        fusion_rlt->InsertOutputs("transpose_1", {0});

        fusion_rlt->SetType(kScopeToMultiNodes);
        std::string scope_name = scopes[0]->Name();
        fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
        fusion_rlt->SetDescription("");

        auto in_identity_0 = fusion_rlt->AddInnerNode("input_identity_0", "Identity");
        CHECK_INNER_NODE_CONDITION(in_identity_0 != nullptr, fusion_rlt);
        Status ret = in_identity_0->InsertInput(kInputFromFusionScope, 0)
                .InsertOutput("inner_core_decode_bbox_v2", 0)
                .BuildInnerNode();
        CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
        std::string str_attr = "input_0_identity_attr";
        in_identity_0->MutableOperator()->SetAttr("key", str_attr);

        auto in_identity_1 = fusion_rlt->AddInnerNode("input_identity_1", "Identity");
        CHECK_INNER_NODE_CONDITION(in_identity_1 != nullptr, fusion_rlt);
        ret = in_identity_1->InsertInput(kInputFromFusionScope, 1)
                .InsertOutput("inner_core_decode_bbox_v2", 1)
                .BuildInnerNode();
        CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);

        auto core_decode_bbox = fusion_rlt->AddInnerNode("inner_core_decode_bbox_v2", kScopeType);
        CHECK_INNER_NODE_CONDITION(core_decode_bbox != nullptr, fusion_rlt);
        ret = core_decode_bbox->InsertInput("input_identity_0", 0)
                .InsertInput("input_identity_1", 0)
                .InsertOutput("output_identity", 0)
                .BuildInnerNode();
        CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);


        auto parser_ret = DecodeBboxV2ParseParams(fusion_rlt->Nodes(), core_decode_bbox->MutableOperator());
        CHECK_INNER_NODE_CONDITION(parser_ret == SUCCESS, fusion_rlt);

        auto out_identity = fusion_rlt->AddInnerNode("output_identity", "Identity");
        CHECK_INNER_NODE_CONDITION(out_identity != nullptr, fusion_rlt);
        ret = out_identity->InsertInput("inner_core_decode_bbox_v2", 0)
                .InsertOutput(kOutputToFusionScope, 0)
                .BuildInnerNode();
        CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);


        ret = fusion_rlt->CheckInnerNodesInfo();
        CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);

        OP_LOGI(kOpType, "Set fusion multi-to-multi result successfully.");
        return;
    }

    REGISTER_SCOPE_FUSION_PASS("DecodeBboxV2MultiScopeFusionPass", DecodeBboxV2MultiScopeFusionPass, false);
}  // namespace ge
