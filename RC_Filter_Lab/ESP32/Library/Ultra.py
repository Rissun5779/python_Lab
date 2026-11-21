from time import sleep, sleep_us
from machine import time_pulse_us, Pin

class ultra:
    def __init__(self, trig: Pin, echo: Pin, timeout_us: int = 30*1000):
        self.trig = trig
        self.echo = echo
        self.timeout_us = timeout_us

        self.trig.value(0)

    def measure_distance(self)->float:
        self.trig.value(0)
        sleep_us(10)

        self.trig.value(1)
        sleep_us(10)
        self.trig.value(0)

        try:
            duration = time_pulse_us(self.echo, 1, self.timeout_us)
        except OSError:
            return -1

        if duration < 0:
            return -1

        return duration*0.0343 / 2