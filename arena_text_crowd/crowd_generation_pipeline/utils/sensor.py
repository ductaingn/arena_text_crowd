import math
import pyDeclutter


class Sensor:
    def __init__(self, agent_r, obs_all, sensorRange=100, N=64):
        self.agent_r = agent_r
        self.obs_all = obs_all
        self.sensorRange = sensorRange
        self.dirs = [
            (math.cos(math.pi * 2 * j / N), math.sin(math.pi * 2 * j / N))
            for j in range(N)
        ]
        self.N = N
        self.reset()

    def reset(self):
        # setup obs
        lss = []
        for obs in self.obs_all:
            for i in range(len(obs)):
                j = (i + 1) % len(obs)
                a = pyDeclutter.Vec2(obs[i][0], obs[i][1])
                b = pyDeclutter.Vec2(obs[j][0], obs[j][1])
                lss.append(pyDeclutter.LineSeg2D(a, b))
        if len(lss) == 0:
            return
        self.envCpp = pyDeclutter.Environment2D(lss)

    def get_sensor_reading(self, positions):
        if len(self.obs_all) == 0:
            self.readings = [
                [max(self.sensorRange - self.agent_r, 0) for si in range(self.N)]
                for ip, p in enumerate(positions)
            ]
            return self.readings
        pss = [pyDeclutter.Vec2(p[0], p[1]) for p in positions]
        self.envCpp.setAgent(pss, self.agent_r)
        self.readings = [
            [
                max(v - self.agent_r, 0)
                for v in self.envCpp.sensorData(p, self.sensorRange, self.N, ip)
            ]
            for ip, p in enumerate(pss)
        ]
        return self.readings
