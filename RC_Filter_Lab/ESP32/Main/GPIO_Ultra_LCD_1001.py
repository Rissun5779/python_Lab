from machine import Pin, I2C, time_pulse_us, UART
from time import sleep, sleep_us
from LCD1602 import lcd1602

trig = Pin(4, Pin.OUT)
echo = Pin(2, Pin.IN)

i2c  = I2C(1, scl=Pin(22), sda=Pin(21), freq=400000)
LCD  = lcd1602(i2c)
devices = i2c.scan()
print("I2C devices:", [hex(k) for k in devices])

LCD.puts("-Ultrasound-", x=2, y=0)
LCD.puts("Distance:", x=0, y=1)
LCD.puts("cm", x=14, y=1)

uart = UART(2, baudrate=9600, tx=17, rx=16, timeout=10)

def measure_distance():
    trig.value(0)
    sleep_us(10)
    
    trig.value(1)
    sleep_us(10)
    trig.value(0)
    
    try:
        duration = time_pulse_us(echo, 1, 30*1000)
    except OSError:
        return -1
    
    if duration < 0:
        return -1
    
    distance = duration*0.0343/2
    return distance

while True:
    distance = measure_distance()
    print("Distance:{:2f} cm".format(distance))
    LCD.puts(str(round(distance,1)), x=9, y=1)
    msg = str(distance) + "\n\r"
    uart.write(msg)
    print("Send:", msg.strip())
    sleep(1)