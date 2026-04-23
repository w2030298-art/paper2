"""
Communication Foundation Library (comm)

通用通信基础组件，支持 MEC、MIMO、OFDM 等场景的模块化复用。

模块:
- channel:     信道模型 (路径损耗, 衰落, MIMO)
- antenna:     天线阵列 (ULA/URA/UPA, 波束赋形, DoA)
- propagation: 传播与链路 (链路预算, SNR/SINR, 移动性)
- signal:      信号模型 (热噪声, AWGN)
- modulation:  调制解调 (QAM)
- queueing:    排队论模型 (M/M/1, M/M/c)
- utils:       工具函数 (dB/W转换)
"""

__version__ = "0.1.0"

# 子模块
from . import channel
from . import antenna
from . import propagation
from . import signal
from . import modulation
from . import queueing
from . import utils
