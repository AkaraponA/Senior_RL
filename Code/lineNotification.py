import requests
from datetime import datetime

class LineNotification():

    def networking():
        url = 'https://notify-api.line.me/api/notify'
        token = 'kYl3pTXfCGqjOg43OzIOlKl2QcBGAsjJma3YytYvstZ'
        headers = {
                    'content-type':
                    'application/x-www-form-urlencoded',
                    'Authorization':'Bearer '+token
                   }
                   
        #curent date and time
        datetimeNow = datetime.now()
        date_time = datetimeNow.strftime("%m/%d/%Y, %H:%M:%S")
        msg = "at time " + date_time

        sendingMessage = requests.post(url, headers=headers , data = {'message':msg})
        print(sendingMessage.text)
