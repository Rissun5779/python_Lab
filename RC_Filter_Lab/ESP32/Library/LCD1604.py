import time
from machine import I2C, Pin

class lcd1604:
    LCD_I2C_ADDR = const(39)   # 39 = 0x27

    def __init__(self, i2c):
        self.i2c = i2c
        self.buf = bytearray(1)
        self.BK = 0x08
        self.RS = 0x00
        self.E = 0x04

        self.setcmd(0x33)
        time.sleep_ms(5)
        self.send(0x30)
        time.sleep_ms(5)
        self.send(0x20)
        time.sleep_ms(5)

        # 4-bit mode, 2-line mode, 5x8 dots
        # 對 16x4 LCD 仍使用 0x28
        self.setcmd(0x28)

        # Display ON, cursor OFF
        self.setcmd(0x0C)

        # Entry mode
        self.setcmd(0x06)

        # Clear display
        self.setcmd(0x01)
        time.sleep_ms(2)

        self.version = '1.0'

    def setReg(self, dat):
        self.buf[0] = dat
        self.i2c.writeto(self.LCD_I2C_ADDR, self.buf)
        time.sleep_ms(1)

    def send(self, dat):
        d = dat & 0xF0
        d |= self.BK
        d |= self.RS

        self.setReg(d)
        self.setReg(d | self.E)
        self.setReg(d)

    def setcmd(self, cmd):
        self.RS = 0
        self.send(cmd)
        self.send(cmd << 4)

    def setdat(self, dat):
        self.RS = 1
        self.send(dat)
        self.send(dat << 4)

    def clear(self):
        self.setcmd(0x01)
        time.sleep_ms(2)

    def backlight(self, on):
        if on:
            self.BK = 0x08
        else:
            self.BK = 0x00
        self.setcmd(0)

    def on(self):
        self.setcmd(0x0C)

    def off(self):
        self.setcmd(0x08)

    def shl(self):
        self.setcmd(0x18)

    def shr(self):
        self.setcmd(0x1C)

    def move_to(self, x, y):
        # 16x4 LCD 常用列位址
        row_addr = [0x80, 0xC0, 0x90, 0xD0]

        if y < 0:
            y = 0
        if y > 3:
            y = 3

        if x < 0:
            x = 0
        if x > 15:
            x = 15

        self.setcmd(row_addr[y] + x)

    def char(self, ch, x=-1, y=0):
        if x >= 0:
            self.move_to(x, y)

        self.setdat(ch)

    def puts(self, s, x=0, y=0):
        # 避免超過 16 字元
        s = str(s)
        if len(s) > 16:
            s = s[:16]

        if len(s) > 0:
            self.char(ord(s[0]), x, y)

            for i in range(1, len(s)):
                self.char(ord(s[i]))

    def put_line(self, s, y):
        # 清除該列後再顯示
        self.puts(" " * 16, 0, y)
        self.puts(s, 0, y)