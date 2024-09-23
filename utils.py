import numpy as np

def Hamm_window(N): #Hamming窗函数，N表示窗口长度
    window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / max(1,(N - 1))) for n in range(N)])
    # window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / max(1,(N - 1))) for n in range(N)])
    window = window/sum(window) #将Hanning窗函数归一化
    return window