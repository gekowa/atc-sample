from te import tik
from te.tik.common.util import reduce_mul, ceil_div, DTYPE_SIZE
from te.platform.cce_conf import te_set_l2_mode
import numpy as np


def matmul_tik_compute(params, kernel_name, new_ws=None, cnt=0):
    te_set_l2_mode(1)
    tik_instance = tik.Tik(tik.Dprofile('v100', 'mini'),
                           err_msg_level=1)
    if not isinstance(params, dict):
        params = params.__dict__
    m, k, n = params['M'], params['K'], params['N']
    data_type = params["data_type"]
    m_tiling_size = int(params["m_tiling_size"])
    n_tiling_size = int(params["n_tiling_size"])
    k_tiling_size = int(params['k_tiling_size'])

    m_cycle_times = params["m_cycle_times"]
    n_cycle_times = params["n_cycle_times"]
    k_cycle_times = params["k_cycle_times"]


    if data_type == "float16":
        C_loc_out_type = "float32"
        K0 = 16
    else:
        C_loc_out_type = "int32"
        K0 = 32
    block_size = 16

    n_thread_num = params['n_thread_num']
    m_thread_num = params['m_thread_num']
    k_thread_num = params['k_thread_num']

    C_gm = tik_instance.Tensor(C_loc_out_type, (n // block_size, m, block_size),
                               name="C_gm", scope=tik.scope_gm)
    A_gm = tik_instance.Tensor(params["data_type"], (k//K0, m, K0), name="A_gm",
                               scope=tik.scope_gm)
    B_gm = tik_instance.Tensor(params["data_type"], (k//K0, n, K0), name="B_gm",
                               scope=tik.scope_gm)


    with tik_instance.for_range(0, 2, block_num=2) as core_id:
        with tik_instance.for_range(0, n_cycle_times//2, thread_num=n_thread_num) as n_idx:
            with tik_instance.for_range(0, m_cycle_times, thread_num=m_thread_num) as m_idx:
                dst_l0c = tik_instance.Tensor(C_loc_out_type, [n_tiling_size//16, m_tiling_size, 16], name='dst_l0c', scope=tik.scope_cbuf_out)
                with tik_instance.for_range(0, k_cycle_times, thread_num=k_thread_num) as k_idx:
                    A_l1 = tik_instance.Tensor(params['data_type'], [k_tiling_size//K0, m_tiling_size, K0], name="A_tiling_l1", scope=tik.scope_cbuf)
                    tik_instance.data_move(A_l1, A_gm[k_idx * k_tiling_size // K0, m_idx*m_tiling_size, :],
                                           0, k_tiling_size//K0, m_tiling_size, m - m_tiling_size, 0)
                    B_l1 = tik_instance.Tensor(params["data_type"], [k_tiling_size//K0, n_tiling_size, K0], name="B_tiling_l1", scope=tik.scope_cbuf)
                    if n-n_tiling_size>65535:
                        with tik_instance.for_range(0, k_tiling_size//K0) as dma_k_idx:
                            tik_instance.data_move(B_l1[dma_k_idx, :, :], B_gm[k_idx*k_tiling_size//K0 + dma_k_idx, (core_id*n_cycle_times//2+n_idx)*n_tiling_size, :],
                                                    0, 1, n_tiling_size, 0, 0)
                    else:
                        tik_instance.data_move(B_l1, B_gm[k_idx*k_tiling_size//K0, (core_id*n_cycle_times//2+n_idx)*n_tiling_size, :], 0, k_tiling_size//K0, n_tiling_size, n-n_tiling_size, 0)
                    with tik_instance.if_scope(k_idx == 0):
                        tik_instance.matmul(dst_l0c, A_l1, B_l1, m_tiling_size, k_tiling_size, n_tiling_size, init_l1out=True)
                    with tik_instance.else_scope():
                        tik_instance.matmul(dst_l0c, A_l1, B_l1, m_tiling_size, k_tiling_size, n_tiling_size, init_l1out=False)
                tik_instance.fixpipe(C_gm[n_tiling_size//16*(core_id*n_cycle_times//2+n_idx), m_idx*m_tiling_size, :], dst_l0c, n_tiling_size//16, m_tiling_size*16*DTYPE_SIZE[C_loc_out_type]//32,
                                     (m-m_tiling_size)*16*DTYPE_SIZE[C_loc_out_type]//32, 0)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=[A_gm, B_gm], outputs=[C_gm])
    return tik_instance


def matmul_tik(input_x1, input_x2, output_y={}, kernel_name="simple_matmul"):
    shape_a = input_x1.get("ori_shape")
    shape_b = input_x2.get("ori_shape")
    m = shape_a[0]
    k = shape_a[1]
    n = shape_b[1]
    data_type = input_x1.get("dtype").lower()
    params = {
        'M': m,
        'K': k,
        'N': n,
        'data_type': data_type,
        'm_tiling_size': 16,
        'm_cycle_times': 1,
        'm_thread_num': 1,
        'n_tiling_size': 64,
        'n_cycle_times': 16,
        'n_thread_num': 1,
        'k_tiling_size': 32,
        'k_cycle_times': 2,
        'k_thread_num': 2
    }
    matmul_tik_compute(params, kernel_name)
