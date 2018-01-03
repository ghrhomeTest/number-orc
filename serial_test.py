import serial
import serial.tools.list_ports
import sys

ser = serial.Serial()
a = serial.tools.list_ports.comports()

print(ser)
print(a[0])