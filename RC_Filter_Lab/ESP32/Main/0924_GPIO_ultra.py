from machine import Pin, I2C, time_pulse_us
from time import sleep, sleep_us
from LCD1602 import lcd1602
from Ultra import ultra

# def measure_distance():
#     trig.value(0)
#     sleep_us(10)

#     trig.value(1)
#     sleep_us(10)
#     trig.value(0)

#     try:
#         duration = time_pulse_us(echo, 1, 30*1000)
#     except OSError:
#         return -1

#     if duration < 0:
#         return -1

#     distance = duration*0.0343 / 2
#     return distance

trig = Pin(4, Pin.OUT) 
echo = Pin(2, Pin.IN)

trig.value(0)
sleep(0.5)

i2c     = I2C(1, scl=Pin(22), sda=Pin(21), freq=400*1000)
display = lcd1602(i2c)
devices = i2c.scan()

print("I2C devices:", [hex(d) for d in devices])

while True:
    distance = measure_distance()

    if distance < 0:
        print("未偵測到障礙物")
    else:
        print("Distance: {:.2f} cm".format(distance))

        if distance <= 20:
            print("!Warning: 前方太近，請停止前進!")
        elif distance <= 50:
            print("!Notice: 前方有障礙物!")
        else:
            print("前方安全")

    print("---------------------")
    sleep(1)