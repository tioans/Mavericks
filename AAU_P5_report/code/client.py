import socket
import struct

# starting socket connection
(HOST,PORT)=('10.8.0.2',2001)
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);
s.connect((HOST,PORT))

# send WAV file
with open('61-70968-0000.wav', 'rb') as f:
	for l in f: 
		s.sendall(l)
print("Message sent to server!")

s.shutdown(socket.SHUT_WR)

# recieve TXT
prediction = s.recv(4096).decode("utf-8")
print("Prediction of message: %s" % prediction) 

s.close()



