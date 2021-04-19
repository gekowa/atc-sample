from __future__ import absolute_import
import numpy as np
from te import tik
from te.tik.common.util import ceil_div, DTYPE_SIZE
from te.platform.cce_conf import te_set_l2_mode


def conv2d_tik_compute(params):
    te_set_l2_mode(1)
    tik_instance = tik.Tik(tik.Dprofile(params["arch"], params["version"]),
                           err_msg_level=1)
    n, c1, h, w, c0 = params["fm_shape"]
    c1, kh, kw, cout, c0 = params["weight_shape"]
    stride_h, stride_w = params["stride_list"]
    dilation_h, dilation_w = params["dilation_list"]
    pad_top, pad_bot, pad_left, pad_right = params["pad_list"]
    kh_dilation = (kh - 1) * dilation_h + 1
    kw_dilation = (kw - 1) * dilation_w + 1
    ho = int(np.ceil((h + pad_top + pad_bot - kh_dilation + 1) / stride_h)) 
    wo = int(np.ceil((w + pad_right + pad_left - kw_dilation + 1) / stride_w))
    round_howo = ceil_div(ho * wo, 16) * 16 

    fm_gm = tik_instance.Tensor(params['fm_dtype'], (n, c1, h, w, c0),
                                name='fm_gm', scope=tik.scope_gm)
    weight_gm = tik_instance.Tensor(params['weight_type'],
                                    (c1, kh, kw, cout, c0), name='weight_gm',
                                    scope=tik.scope_gm)

    if params['dst_gm_type'] in ("int8", "uint8"):
        dst_gm = tik_instance.Tensor(params['dst_gm_type'],
                                     [n, cout // 32, ho, wo, 32],
                                     name='dst_gm', scope=tik.scope_gm)
    else:
        dst_gm = tik_instance.Tensor(params['dst_gm_type'],
                                     [n, cout // 16, ho, wo, 16],
                                     name='dst_gm', scope=tik.scope_gm)

    core_num = 2
    pre_core_cout = cout // core_num
    cout_iter_num = pre_core_cout // params["cout_split_factor"]
    Cin_blocks = c1

    with tik_instance.for_range(0, core_num, block_num=core_num) as cout_o:
        with tik_instance.for_range(0, cout_iter_num, thread_num=1) as cout_i:
            weight_L1 = tik_instance.Tensor(
                params['weight_type'], (Cin_blocks, kh, kw,
                                        params["cout_split_factor"], c0),
                name='weight_l1', scope=tik.scope_cbuf)
            tik_instance.data_move(
                weight_L1,
                weight_gm.flatten()[cout_o * pre_core_cout * c0 +
                                    params["cout_split_factor"] * cout_i * c0],
                0, Cin_blocks * kh * kw, params["cout_split_factor"], (cout - params["cout_split_factor"]), 0)

            with tik_instance.for_range(0, n, thread_num=2) as n_index:
                feature_map_l1 = tik_instance.Tensor(params['fm_dtype'],
                                                     (c1, h, w, c0),
                                                     name='feature_map_l1',
                                                     scope=tik.scope_cbuf)
                tik_instance.data_move(feature_map_l1,
                                        fm_gm[n_index, :, :, :, :],
                                        0, 1, c1 * h * w, 0, 0)
                dst_l0c = tik_instance.Tensor(
                    params['dst_l0c_type'], [params["cout_split_factor"]//16,
                                             round_howo, 16],
                    name='dst_l0c', scope=tik.scope_cbuf_out)

                tik_instance.conv2d(dst_l0c, feature_map_l1,
                                    weight_L1, (c1, h, w, c0),
                                    (Cin_blocks, kh, kw,
                                     params["cout_split_factor"], c0),
                                    params['stride_list'],
                                    params['pad_list'],
                                    params['dilation_list'],
                                    params['pad_value'])

                tik_instance.fixpipe(
                    dst_gm[n_index, (cout_o*pre_core_cout +
                                     params["cout_split_factor"]*cout_i) //
                           (32//DTYPE_SIZE[params['dst_gm_type']]), 0, 0, 0],
                    dst_l0c, params["cout_split_factor"]//16,
                    ho * wo * 16 * DTYPE_SIZE[params['dst_l0c_type']] // 32, 0, 0,
                    extend_params={"bias": None,
                                   "quantize_params": params["quantize_params"]})

    tik_instance.BuildCCE(kernel_name=params["kernel_name"],
                          inputs=[fm_gm, weight_gm], outputs=[dst_gm])

    return tik_instance


def conv2d_tik(inputs, weights, outputs, strides, pads, dilations, kernel_name="conv2d_tik"):
    in_dtype = inputs.get("dtype")
    w_dtype = weights.get("dtype")
    res_dtype = outputs.get("dtype")
    in_shape = inputs.get("shape")
    wori_shape = weights.get("ori_shape")
            
    if len(strides) != 4:
        raise RuntimeError("strides shape should be 4d.")
    if len(dilations) != 4:
        raise RuntimeError("dilations shape should be 4d.")
    if len(pads) != 4:
        raise RuntimeError("pads shape should be 4d.")
        
    if in_dtype=="float16":
        loc_dtype = "float32"
        quantize_params = {"mode":"fp322fp16", "mode_param":None}
        if weights.get("ori_format")=="NCHW":
            strideList = [strides[2], strides[3]]
            dilationList = [dilations[2], dilations[3]]                    
            w_shape = [wori_shape[1]//16, wori_shape[2], wori_shape[3], wori_shape[0], 16]
        else:
            strideList = [strides[1], strides[2]]
            dilationList = [dilations[1], dilations[2]]                    
            w_shape = [wori_shape[3]//16, wori_shape[1], wori_shape[2], wori_shape[0], 16]
    elif in_dtype=="int8":
        loc_dtype = "int32"
        quantize_params = {"mode":"int322fp16", "mode_param":1.0}
        if weights.get("ori_format")=="NCHW":
            strideList = [strides[2], strides[3]]
            dilationList = [dilations[2], dilations[3]] 
            w_shape = [wori_shape[1]//32, wori_shape[2], wori_shape[3], wori_shape[0], 32]
        else:
            strideList = [strides[1], strides[2]]
            dilationList = [dilations[1], dilations[2]]     
            w_shape = [wori_shape[3]//32, wori_shape[1], wori_shape[2], wori_shape[0], 32]
    else:
         raise RuntimeError("input_dtype shape should be float16 or int8.")

    params = {
        "arch": "v100",
        "version": "mini",
        "fm_shape": in_shape,
        "weight_shape": w_shape,
        "fm_dtype": in_dtype,
        "weight_type": w_dtype,
        "dst_l0c_type": loc_dtype,
        "dst_gm_type": res_dtype,
        "quantize_params": quantize_params,
        "pad_list": pads,
        "pad_value": 0,
        "stride_list": strideList,
        "dilation_list": dilationList,
        "cout_split_factor": 64,
        "kernel_name": kernel_name}

    conv2d_tik_compute(params)
