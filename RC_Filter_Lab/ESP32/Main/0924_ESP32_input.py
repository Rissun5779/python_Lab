from machine import Pin, I2C
from time import sleep
from LCD1602 import lcd1602

sw2 = Pin(13, Pin.IN)
sw1 = Pin(36, Pin.IN)

i2c     = I2C(1, scl=Pin(22), sda=Pin(21), freq=400*1000)
display = lcd1602(i2c)
devices = i2c.scan()

print("I2C devices:", [hex(d) for d in devices])

while True:
    if sw1.value() == 0:
        print("sw1 D36Pin 按鍵被壓")
        display.puts("SW1D36 pull down", x=0, y=0)
    else:
        print("sw1 D36Pin 按鍵被鬆開")
        display.puts("SW1D36 pull -up-", x=0, y=0)
    
    if sw2.value() == 0:
        print("sw2 D13Pin 按鍵被壓")
        display.puts("SW2D13 pull down", x=0, y=1)
    else:
        print("sw2 D13Pin 按鍵被鬆開")
        display.puts("SW2D13 pull -up-", x=0, y=1)
    sleep(1)