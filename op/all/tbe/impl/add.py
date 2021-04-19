#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

add
"""
from __future__ import absolute_import

from functools import reduce
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic

# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648


def _produce_shapes(shape1, shape2):
    """
    two input shapes produce three output shape
    """
    shape1 = list(shape1)
    shape2 = list(shape2)
    flag = 0
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
        flag = 1

    output_shape_len = len(shape1)
    dec = output_shape_len - len(shape2)
    for i in range(dec):
        shape2 = [1] + shape2

    out_shape = []
    for i in range(output_shape_len):
        if (shape1[i] != shape2[i]) and (shape1[i] != 1) and (shape2[i] != 1):
            raise RuntimeError("input shapes not match!")
        out_shape.append(shape1[i] if shape1[i] > shape2[i] else shape2[i])

    if flag == 1:
        shape1, shape2 = shape2, shape1

    return shape1, shape2, out_shape


def _shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    result = []
    for i in shape:
        if isinstance(i, tvm.expr.Var):
            result.append(i)
        else:
            result.append(i.value)
    return result


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("add")
def add_compute(input_x, input_y, output_z, kernel_name="add"):
    """
    calculating data's add, c = a + b

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    input_y: TVM tensor
        the placeholder of second input data
    output_data: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is add

    Returns
    -------
    res : output of the data's add
    """
    shape_x = _shape_to_list(input_x.shape)
    shape_y = _shape_to_list(input_y.shape)

    shape_x, shape_y, shape_max = _produce_shapes(shape_x, shape_y)
    shape_size = reduce(lambda x, y: x * y, shape_max[:])
    if shape_size > SHAPE_SIZE_LIMIT:
        raise RuntimeError("the shape is too large to calculate")

    input_x = te.lang.cce.broadcast(input_x, shape_max)
    input_y = te.lang.cce.broadcast(input_y, shape_max)
    res = te.lang.cce.vadd(input_x, input_y)

    return res


def add(input_x, input_y, output_z, kernel_name="add"):
    """
    algorithm: add
    calculating data's add, c = a + b

    Parameters
    ----------
    input_x : dict
        shape and dtype of first input, only support float16, float32, int32
    input_y : dict
        shape and dtype of second input, only support float16, float32, int32
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name : str
        cce kernel name, default value is add

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")

    check_tuple = ("float16", "float32", "int32")
    input_data_type = input_x.get("dtype").lower()
    if input_data_type not in check_tuple:
        raise RuntimeError("only support %s while dtype is %s" %
                           (",".join(check_tuple), input_data_type))

    shape_x, shape_y, shape_max = _produce_shapes(shape_x, shape_y)
    if shape_x[-1] == 1 and shape_y[-1] == 1 and shape_max[-1] == 1:
        shape_x = shape_x if len(shape_x) == 1 else shape_x[:-1]
        shape_y = shape_y if len(shape_y) == 1 else shape_y[:-1]
        shape_max = shape_max if len(shape_max) == 1 else shape_max[:-1]

    data_x = tvm.placeholder(shape_x, name="data_1", dtype=input_data_type)
    data_y = tvm.placeholder(shape_y, name="data_2", dtype=input_data_type)

    res = add_compute(data_x, data_y, output_z, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_x, data_y, res)}
    te.lang.cce.cce_build_code(schedule, config)
