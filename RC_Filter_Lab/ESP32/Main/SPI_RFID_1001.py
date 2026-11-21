from machine import Pin, I2C, SPI
from time import sleep
from LCD1602 import lcd1602
from MFRC522 import mfrc522

i2c  = I2C(1, scl=Pin(22), sda=Pin(21), freq=400000)
LCD  = lcd1602(i2c)
devices = i2c.scan()
print("I2C devices:", [hex(k) for k in devices])

LCD.puts("RFID UID CODE", x=1, y=0)

sck = Pin(18, Pin.OUT)
mosi = Pin(23, Pin.OUT)
miso = Pin(19, Pin.OUT)
sda = Pin(5, Pin.OUT)
spi = SPI(baudrate=100*1000, polarity=0, phase=0,sck=sck, mosi=mosi, miso=miso)
sw2 = Pin(13, Pin.IN)

while True:
    if sw2.value()==0:
        print("Sensoring sw2 be push down")
        mfrc = mfrc522(spi, sda)
        uid=""
        (stat, tag_type) = mfrc.request(mfrc.REQIDL)
        if stat == mfrc.OK:
            (stat, raw_uid) = mfrc.anticoll()
            if stat == mfrc.OK:
                uid = ("0x%02x%02x%02x%02x"%(raw_uid[0], raw_uid[1], raw_uid[2], raw_uid[3]))
                print("RFID UID:", uid)
                LCD.puts(uid, x=0, y=1)
        sleep(2)