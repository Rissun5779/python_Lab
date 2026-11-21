from machine import Pin
import time
import network
from umqtt.robust import MQTTClient
import dht

client = MQTTClient(
    client_id="temp", 
    server="io.adafruit.com", 
    user="tsaipusheng", 
    password="aio_UZhw32bfjJ8XvP4ZO18T56yhIq7G",ssl=False)

sta_if = network.WLAN(network.STA_IF)     # 取得無線網路介面
sta_if.active(True)                       # 啟用無線網路
sta_if.connect("tps", "tsaipusheng1021")
while not sta_if.isconnected():           # 等待無線網路連上
    pass

print("已經連上手機IP")

client.connect()

shock = Pin(13, Pin.IN)
sensor = dht.DHT11(Pin(14))
while True:
    if shock.value() == 0:
        print("感應到按鍵動作發生!")

        sensor.measure()
        temp_humi="%2d℃/%2d%%" (sensor.temperature(),sensor.humidity())

        #client.publish(b"tsaipusheng/feeds/temp",temp_humi.encode())
        print(temp_humi)
        time.sleep(3)
