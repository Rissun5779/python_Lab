from machine import Pin, I2C, time_pulse_us, UART, ADC
from time import sleep, sleep_us
from LCD1602 import lcd1602

adc = ADC(Pin(34))
uart = UART(2, baudrate=9600, tx=17, rx=16, timeout=10)

i2c  = I2C(1, scl=Pin(22), sda=Pin(21), freq=400000)
LCD  = lcd1602(i2c)
devices = i2c.scan()
print("I2C devices:", [hex(k) for k in devices])

adc.atten(ADC.ATTN_11DB)
adc.width(ADC.WIDTH_12BIT)

sample_count = 1000
sample_time  = 0.01

uart.write("START\n\r""-------------------\n\r")
print("Start transfer number 1000 data")

data_list = []

LCD.puts("ADC Value:", x=0, y=0)
LCD.puts("ADC Valte:    V", x=0, y=1)

for k in range(sample_count):
    value = adc.read()
    data_list.append(value)
    msg = "{},{}\n\r".format(k, value)
    uart.write(msg)
    print(msg.strip())
    sleep(sample_time)
    
uart.write("END\n\r")

