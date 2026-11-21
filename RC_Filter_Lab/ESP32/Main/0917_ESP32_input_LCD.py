from machine import Pin
from time import sleep
from LCD1602 import lcd1602

sw1 = Pin(36, Pin.IN)
sw2 = Pin(13, Pin.IN)

while (1):
	if sw2.value()==0:
		print("sw2 pressed", sw2.value())
	else:
		print("sw2 release", sw2.value())
	
	if sw1.value()==0:
		print("sw1 pressed", sw1.value())
	else:
		print("sw1 release", sw1.value())
	sleep(1)

