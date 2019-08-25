from gpuinfo.nvidia import get_gpus
import psutil
import os

class hardwareUsage:
    def __init__(self):
        pass

    def CPU(self):
        CPU_Usage = float(psutil.cpu_percent())
        return CPU_Usage

    def RAM(self):
        return psutil.virtual_memory()[2]

    def GPU(self):
        gpu_usage = []
        for gpu in get_gpus():
            gpu_usage.append(gpu.get_clock_speeds(0)/gpu.get_max_clock_speeds(0))
        Calc_Power = sum(gpu_usage)
        return Calc_Power
        
    def Pred_E(self):
        E = self.RAM()+self.CPU()
        return E
    


        

