import sys
import time
import math
from machine import Pin, DAC, ADC

dac = DAC(Pin(25))
adc_in = ADC(Pin(34))
adc_out = ADC(Pin(35))
adc_in.atten(ADC.ATTN_11DB)
adc_out.atten(ADC.ATTN_11DB)

LUT_SIZE = 256
SIN_LUT = bytearray(int(128 + 39 * math.sin(2 * math.pi * i / LUT_SIZE)) for i in range(LUT_SIZE))

while True:
    # 等待 PC 發送量測指令: "頻率,是否方波,取樣間隔(us),總點數"
    line = sys.stdin.readline().strip()
    if not line or ',' not in line:
        continue
    parts = line.split(',')
    freq = float(parts[0])
    is_square = (parts[1] == '1')
    sample_us = int(parts[2])
    n_samples = int(parts[3])

    # 先空跑 0.3 秒讓 RC 電路進入穩態
    t0 = time.ticks_us()
    while time.ticks_diff(time.ticks_us(), t0) < 300_000:
        dt = time.ticks_diff(time.ticks_us(), t0)
        ph = ((dt * freq) / 1_000_000.0) % 1.0
        dac.write(((128 + 39) if ph < 0.5 else (128 - 39)) if is_square else SIN_LUT[int(ph * LUT_SIZE)])

    # 高速無阻塞取樣進記憶體 (不受 UART 速度限制)
    buf_t = [0] * n_samples
    buf_in = [0] * n_samples
    buf_out = [0] * n_samples

    t_start = time.ticks_us()
    last_t = t_start
    idx = 0
    while idx < n_samples:
        now = time.ticks_us()
        if time.ticks_diff(now, last_t) >= sample_us:
            last_t = time.ticks_add(last_t, sample_us)
            dt = time.ticks_diff(now, t_start)
            ph = ((dt * freq) / 1_000_000.0) % 1.0
            dac.write(((128 + 39) if ph < 0.5 else (128 - 39)) if is_square else SIN_LUT[int(ph * LUT_SIZE)])
            
            buf_t[idx] = dt
            buf_in[idx] = adc_in.read()
            buf_out[idx] = adc_out.read()
            idx += 1

    # 取樣結束後一次性回傳給 PC
    sys.stdout.write("START\n")
    for i in range(n_samples):
        sys.stdout.write(f"{buf_t[i]},{buf_in[i]},{buf_out[i]}\n")
    sys.stdout.write("END\n")