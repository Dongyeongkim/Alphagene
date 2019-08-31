import psutil


class hardwareUsage:
    def __init__(self):
        pass

    def CPU(self):
        CPU_Usage = float(psutil.cpu_percent())
        return CPU_Usage

    def RAM(self):
        return psutil.virtual_memory()[2]

    '''def GPU(self):
        for gpu in get_gpus():
            max_clock_speed = gpu.get_max_clock_speeds()
            current_clock_speed = gpu.get_clock_speeds()
            Core_Usage = (current_clock_speed['core_clock_speed'] / max_clock_speed['max_core_clock_speed'])
            Memory_Usage = (current_clock_speed['memory_clock_speed'] / max_clock_speed['max_memory_clock_speed'])
        return 100 * (Core_Usage + Memory_Usage)'''

    def Pred_E(self):
        E = self.RAM() + self.CPU()+'''+self.GPU()'''
        return E

