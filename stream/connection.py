import websockets
import numpy as np
import asyncio
import sounddevice as sd
import sys
import json

class stream:

    HOST = "165.22.80.252"
    PORT = 8081
    cnt = 0

    def __init__(self):
        self.uri_audio = "ws://" + self.HOST + ":"+str(self.PORT)+"/audio"
        self.uri_data = "ws://localhost:"+str(self.PORT)+"/data"
    
    async def send_audio(self,is_speech):
        try:
            async with websockets.connect(self.uri_audio) as ws:
                await ws.send(bytes(str(json.dumps(is_speech)),'utf-8'))
                return
        except Exception as e:
            print(type(e).__name__ + str(e))
            while True:
                print("Try to reconnect or not?\ny/n\n")
                key = input()
                if key == 'y':
                    self.send_audio(is_speech)
                    break
                if key == 'n':
                    print("Program was stopped")
                    sys.exit()
                else:
                    print("Input y or n please")
                    continue
    
    async def send_data(self,data):
        try:
            async with websockets.connect(self.uri_data) as ws:
                await ws.send(bytes(str(data),'utf-8'))
        except Exception as e:
            print(type(e).__name__ + str(e))
            while True:
                print("Try to reconnect or not?\ny/n\n")
                key = input()
                if key == 'y':
                    self.send_data(data)
                    break
                if key == 'n':
                    print("Program was stopped")
                    sys.exit()
                else:
                    print("Input y or n please")
                    continue
