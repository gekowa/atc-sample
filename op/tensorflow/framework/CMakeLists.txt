# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/caffe_plugin")
    add_subdirectory(caffe_plugin)
endif()
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tf_plugin")
    add_subdirectory(tf_plugin)
endif()
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tf_scope_fusion_pass")
    add_subdirectory(tf_scope_fusion_pass)
endif()