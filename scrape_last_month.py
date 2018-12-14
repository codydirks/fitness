import pandas as pd
import fitbit
import gather_keys_oauth2 as Oauth2
import datetime as dt
import time
import sys

# For now, I'm storing data as the raw json that gets
# retrieved through the API, and I'll figure out later
# in the analysis how to parse this out into separate
# time series data and summary/metadata
import json

# Gets authorization credentials - will pop open a browser
# window to get user authorization
keys=json.loads(open('api_keys.json','r').read())
CLIENT_ID,CLIENT_SECRET=[keys[i] for i in ('fitbit-client-id','fitbit-client-secret')]
server = Oauth2.OAuth2Server(CLIENT_ID, CLIENT_SECRET)
server.browser_authorize()
ACCESS_TOKEN = str(server.fitbit.client.session.token['access_token'])
REFRESH_TOKEN = str(server.fitbit.client.session.token['refresh_token'])

# Creates the main Fitbit API object that we'll call to get data
client = fitbit.Fitbit(CLIENT_ID, CLIENT_SECRET, oauth2=True, access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN)

# Retrieve data for past x days
if len(sys.argv)==1:
    x=30
else:
    x=int(sys.argv[1])
days=[(((dt.datetime.now() - dt.timedelta(days=i)).strftime("%Y-%m-%d"))) for i in reversed(range(x))]

#Retrieve HR and sleep data
for day in days:
    # Retrieves and stores heart rate data json for given day
    hr_json = client.intraday_time_series('activities/heart', base_date=day, detail_level='1sec')
    sleep_json=client.sleep(date=day)
    hr_file='data/heartrate/{:}_hr.json'.format(day)
    sleep_file='data/sleep/{}_sleep.json'.format(day)
    for file,js in [(hr_file,hr_json),(sleep_file,sleep_json)]:
        print('Saving to {}'.format(file))
        with open(file,'w') as fp:
            json.dump(js,fp,indent=4)

#Retrieve and parse activity data into individual days
url=''.join(['https://api.fitbit.com/1/user/-/activities/list.json?afterDate=',
             str(dt.datetime.strptime(days[0], "%Y-%m-%d").date()-dt.timedelta(days=1)),
             '&sort=asc&offset=0&limit=100'])
all_acts=client.client.session.request('GET',url)
all_jsons=json.loads(all_acts.content.decode('utf8'))

for day in days:
    acts=[i for i in all_jsons['activities'] if i['startTime'].split('T')[0]==day]

    # Once we isolate activities for this day, send requests for the detailed heart rate
    # data for each activity
    acts_json=[]
    for act in acts:
        acts_request=client.client.session.request('GET',act['heartRateLink'])
        acts_json.append(json.loads(acts_request.content.decode('utf-8')))

    #Store the resulting day's worth of activities as a JSON list
    if len(acts_json)>0:
        act_file='data/activities/{:}_acts.json'.format(day)
        print('Saving to {}'.format(act_file))
        with open(act_file,'w') as fp:
            json.dump(acts_json,fp,indent=4)
