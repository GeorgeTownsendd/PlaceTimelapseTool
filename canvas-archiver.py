import os
import requests
import time
import json
from requests.auth import HTTPBasicAuth
import websocket
from PIL import Image
from datetime import datetime
from authparams import USERNAME, PASSWORD, OAUTH_CLIENT, OAUTH_SECRET
from typing import Dict, Optional
import re

class Archiver:
    def __init__(self):
        self.auth_token = None
        self.current_config: Dict[int, Dict] = {}
        self.base_dir = "canvas_data/place_2023/canvas/"
        os.makedirs(self.base_dir, exist_ok=True)
        self.fetched_files = {}
        self.timestamp = None

    def auth(self):
        try:
            r = requests.post(f'https://www.reddit.com/api/v1/access_token?grant_type=password&username={USERNAME}&password={PASSWORD}',
                            auth=HTTPBasicAuth(OAUTH_CLIENT, OAUTH_SECRET))
            auth_response = json.loads(r.text)
            if 'access_token' in auth_response:
                return auth_response['access_token']
            else:
                print("Access token not found in the auth response")
                return None
        except Exception as e:
            print(f"Error during auth: {e}")
            return None

    def get_auth_token(self):
        while True:
            self.auth_token = self.auth()
            if self.auth_token is not None:
                break
            time.sleep(5)

    def on_message(self, ws, message):
        payload = json.loads(message)
        print(f"Received payload: {payload}")  # Debug statement
        if payload['type'] == "connection_error":
            ws.close()
            return

        if payload['type'] != "data":
            return

        if payload['payload']['data']['subscribe']['data']['__typename'] == "ConfigurationMessageData":
            messageIndex = 2
            canvasConfig = payload['payload']['data']['subscribe']['data']['canvasConfigurations']
            canvasHeight = payload['payload']['data']['subscribe']['data']['canvasHeight']
            canvasWidth = payload['payload']['data']['subscribe']['data']['canvasWidth']
            activeZoneRaw = payload['payload']['data']['subscribe']['data']['activeZone']
            activeZone = {
                "startX": activeZoneRaw['topLeft']['x'],
                "startY": activeZoneRaw['topLeft']['y'],
                "endX": activeZoneRaw['bottomRight']['x'],
                "endY": activeZoneRaw['bottomRight']['y']
            }
            for configItem in canvasConfig:
                if configItem['__typename'] == "CanvasConfiguration":
                    itemIndex = configItem['index']
                    self.current_config[itemIndex] = {
                        "url": None,
                        "completed": False,
                        "startX": configItem['dx'],
                        "startY": configItem['dy'],
                        "endX": configItem['dx'] + canvasWidth,
                        "endY": configItem['dy'] + canvasHeight
                    }
                    if (self.current_config[itemIndex]["endX"] <= activeZone["startX"] or
                            self.current_config[itemIndex]["startX"] >= activeZone["endX"]):
                        self.current_config[itemIndex]['completed'] = True
                    if (self.current_config[itemIndex]["endY"] <= activeZone["startY"] or
                            self.current_config[itemIndex]["startY"] >= activeZone["endY"]):
                        self.current_config[itemIndex]['completed'] = True
            for index in self.current_config.keys():
                if (not self.current_config[index]['completed']):
                    ws.send('{"id":"' + str(
                        messageIndex) + '","type":"start","payload":{"variables":{"input":{"channel":{"teamOwner":"GARLICBREAD","category":"CANVAS","tag":"' + str(
                        index) + '"}}},"extensions":{},"operationName":"replace","query":"subscription replace($input:SubscribeInput!){subscribe(input:$input){id...on BasicMessage{data{__typename...on FullFrameMessageData{__typename name timestamp}...on DiffFrameMessageData{__typename name currentTimestamp previousTimestamp}}__typename}__typename}}"}}')
                    messageIndex += 1

        if payload['payload']['data']['subscribe']['data']['__typename'] == "FullFrameMessageData":
            url = payload['payload']['data']['subscribe']['data']['name']
            extractedIndex = int(re.search("[0-9]{13}-([0-9]{1})", url).group(1))
            self.current_config[extractedIndex]['url'] = url
            self.fetch_image_from_url(url, extractedIndex)
            if all([config['completed'] for config in self.current_config.values()]):
                ws.close()


    def get_directory_name(self):
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        directory_name = os.path.join(self.base_dir, self.timestamp)
        os.makedirs(directory_name, exist_ok=True)
        return directory_name

    def on_open(self, ws):
        print("Connection opened")
        self.directory_name = self.get_directory_name()  # Create the directory here
        ws.send('{"type":"connection_init","payload":{"Authorization":"Bearer ' + self.auth_token + '"}}')
        ws.send('{"id":"1","type":"start","payload":{"variables":{"input":{"channel":{"teamOwner":"GARLICBREAD","category":"CONFIG"}}},"extensions":{},"operationName":"configuration","query":"subscription configuration($input:SubscribeInput!){subscribe(input:$input){id...on BasicMessage{data{__typename...on ConfigurationMessageData{colorPalette{colors{hex index __typename}__typename}canvasConfigurations{index dx dy __typename}activeZone{topLeft{x y __typename}bottomRight{ x y __typename} __typename}canvasWidth canvasHeight __typename}}__typename}__typename}}"}}')

    def combine_and_save(self):
        final_image = Image.new('RGBA', (3000, 2000))  # Create a new image with transparency
        for idx, config in sorted(self.current_config.items()):
            if config['completed'] and idx in self.fetched_files:
                img = Image.open(self.fetched_files[idx])  # Retrieve the filename
                final_image.paste(img, (config['startX'], config['startY']))
        final_image.save(os.path.join(self.directory_name, f"{self.timestamp}.png"), "PNG")  # Save the final image in PNG format to retain transparency


    def fetch_image_from_url(self, url, index):
        response = requests.get(url)
        filename = os.path.join(self.directory_name, f"{index}-{int(time.time())}.png")
        with open(filename, 'wb') as file:
            file.write(response.content)
        self.current_config[index]['completed'] = True
        self.fetched_files[index] = filename  # Save the filename

    def on_open(self, ws):
        print("Connection opened")
        self.directory_name = self.get_directory_name()  # Create the directory here
        ws.send('{"type":"connection_init","payload":{"Authorization":"Bearer ' + self.auth_token + '"}}')
        ws.send('{"id":"1","type":"start","payload":{"variables":{"input":{"channel":{"teamOwner":"GARLICBREAD","category":"CONFIG"}}},"extensions":{},"operationName":"configuration","query":"subscription configuration($input:SubscribeInput!){subscribe(input:$input){id...on BasicMessage{data{__typename...on ConfigurationMessageData{colorPalette{colors{hex index __typename}__typename}canvasConfigurations{index dx dy __typename}activeZone{topLeft{x y __typename}bottomRight{ x y __typename} __typename}canvasWidth canvasHeight __typename}}__typename}__typename}}"}}')

    def run(self):
        self.get_auth_token()
        while True:
            try:
                ws = websocket.WebSocketApp("wss://gql-realtime-2.reddit.com/query",
                                            on_message=self.on_message,
                                            on_open=self.on_open)
                ws.run_forever()
            except Exception as ex:
                self.get_auth_token()
                print("ERROR")
                print(str(ex))
            self.combine_and_save()
            time.sleep(30)

    def run(self):
        self.get_auth_token()
        while True:
            try:
                ws = websocket.WebSocketApp("wss://gql-realtime-2.reddit.com/query",
                                            on_message=self.on_message,
                                            on_open=self.on_open)
                ws.run_forever()
            except Exception as ex:
                self.get_auth_token()
                print("ERROR")
                print(str(ex))
            self.combine_and_save()
            time.sleep(30)

if __name__ == "__main__":
    archiver = Archiver()
    archiver.run()
