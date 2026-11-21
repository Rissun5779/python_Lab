from machine import UART
from time import sleep

uart = UART(2, baudrate=9600, tx=17, rx=16, timeout=10)

while True:
    msg = "Ming Chun University" + "\n\r"
    uart.write(msg)
    print("Send:", msg.strip())
    sleep(1)
