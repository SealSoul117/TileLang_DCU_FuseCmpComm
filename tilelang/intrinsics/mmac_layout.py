from tvm import DataType
from tvm.runtime import convert
import tilelang.language as T


'''
256x16 A share mem layout
  0   16   32   48
  1   17   33   49
  2   18   34   50
  3   19   35   51
---------------------
  4   20   36   52
  5   21   37   53


16x16 A Reg  layout
    0  16  32  48
    1  17  33  49
    2  18  34  50
    3  19  35  51
'''

def shared_16x16_to_local_64x4_layout_A(i, j):
    # i: row (0-15), j: column (0-15)
    # thread_id: which thread handles this position
    # local_id: which element within the thread's 4-element vector
    #
    # Layout (根据你的图示):
    #   j=0-3      j=4-7      j=8-11     j=12-15
    #   T0-T3      T16-T19    T32-T35    T48-T51   (i=0)
    #   T4-T7      T20-T23    T36-T39    T52-T55   (i=1)
    #   T8-T11     T24-T27    T40-T43    T56-T59   (i=2)
    #   T12-T15    T28-T31    T44-T47    T60-T63   (i=3)
    thread_id = i + 16 * (j // 4)  # 0-63
    local_id = j % 4                # 0-3: 4 consecutive elements per thread
    return thread_id, local_id

def thread_id_shared_access_64x4_to_16x16_layout_A(tx, local_id):
    # tx: thread id within warp (0-63)
    # local_id: element index within thread's 4-element vector (0-3)
    #
    # 根据布局:
    #   tx=0-15  处理行 0-15 的同一位置，列块 0 (j=0-3)
    #   tx=16-31 处理行 0-15 的同一位置，列块 1 (j=4-7)
    #   tx=32-47 处理行 0-15 的同一位置，列块 2 (j=8-11)
    #   tx=48-63 处理行 0-15 的同一位置，列块 3 (j=12-15)
    #
    # 例如 tx=0, local_id=0 → (i=0, j=0)
    #     tx=0, local_id=1 → (i=0, j=1)
    #     tx=16, local_id=0 → (i=0, j=4)
    i = tx % 16                  # 0-15: row position within the 16-row block
    j = (tx // 16) * 4 + local_id  # 0-15: column position
    return i, j

shared_16x16_to_local_64x4_layout_m_n = shared_16x16_to_local_64x4_layout_A
shared_16x16_to_local_64x4_layout_n_k = shared_16x16_to_local_64x4_layout_A