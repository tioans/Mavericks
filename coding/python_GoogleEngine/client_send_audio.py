import speech_recognition as sr
import socket
import struct

# obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

# write audio to a WAV file
with open("file_to_send.wav", "wb") as f:
    f.write(audio.get_wav_data())


(HOST,PORT)=('10.8.0.2',2001)
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);
s.connect((HOST,PORT))

# send WAV file
with open('file_to_send.wav', 'rb') as f:
	for l in f: 
		s.sendall(l)

s.shutdown(socket.SHUT_WR)

# recieve TXT
prediction = s.recv(4096).decode("utf-8") 
print (prediction)

s.close()