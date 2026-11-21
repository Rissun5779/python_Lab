from machine import Pin, I2C, time_pulse_us, UART, ADC
from time import sleep, sleep_us
from LCD1602 import lcd1602

adc = ADC(Pin(34))

i2c  = I2C(1, scl=Pin(22), sda=Pin(21), freq=400000)
LCD  = lcd1602(i2c)
devices = i2c.scan()
print("I2C devices:", [hex(k) for k in devices])

adc.atten(ADC.ATTN_11DB)
adc.width(ADC.WIDTH_12BIT)

LCD.puts("ADC Value:", x=0, y=0)
LCD.puts("ADC Valte:    V", x=0, y=1)

while True:
    value = adc.read()
    LCD.puts(str(value), x=10, y=0)
    print("ADC convert value:", value)
    voltage = round(((3.3*value)/4096), 1)
    print("ADC convert voltage:", voltage, "V")
    LCD.puts(str(voltage), x=10, y=1)
    sleep(1)