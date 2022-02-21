import serial

ser = serial.Serial('COM3')

ser.write(b'a')

while True:
	print(ser.read())