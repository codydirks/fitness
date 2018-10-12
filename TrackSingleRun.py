import sys
import requests

import numpy as np
from math import log, exp, tan, atan, ceil

from PIL import Image
from io import BytesIO

#Matplotlib imports
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.basemap import Basemap

import json
with open('api_keys.json','r') as key_file:
    keys=json.loads(key_file.read())
    GOOGLE_MAPS_API_KEY,GOOGLE_MAPS_SECRET=[keys[i] for i in ('google-maps-api-key','google-maps-client-secret')]
    STRAVA_CLIENT_ID,STRAVA_CLIENT_SECRET,STRAVA_ACCESS_TOKEN=[keys[i]
                                                           for i in ('strava-client-id','strava-client-secret','strava-access-token')]
GOOGLE_API_URL = 'https://maps.googleapis.com/maps/api/staticmap?'

# Some constants

# circumference/radius
tau = 6.283185307179586
# One degree in radians, i.e. in the units the machine uses to store angle,
# which is always radians. For converting to and from degrees. See code for
# usage demonstration.
DEGREE = tau/360

ZOOM_OFFSET = 8

# Max width or height of a single image grabbed from Google.
# For Google's free access plan, this is the largest possible
# image that can be requested, so probably best to leave this
# be.
MAXSIZE = 600

# For cutting off the logos at the bottom of each of the grabbed images.  The
# logo height in pixels is assumed to be less than this amount.
LOGO_CUTOFF = 25

# Width and height of entire map in METERS.
# This should be large enough to encompass all the expected data
basemap_width=30000
basemap_height=30000

# Subset of basemap dimensions to focus on
# Use this to tune your output region a bit
# (0,0) in these coordinates would be the SW corner of the map
xmin,xmax=10000,19500
ymin,ymax=0,23650


# Define functions for obtaining individual static Google Maps images and
# stitching them together. Must run this cell first.
# Taken from https://stackoverflow.com/a/50536888
import hashlib
import hmac
import base64
import urllib.parse as urlparse

# Straightforward functions to converb b/w
# a Google Maps image's pixel coordinates
# and lat/lon coordinates. Requires
# the Google Maps 'zoom' level parameter as
# input.
def latlon2pixels(lat, lon, zoom):
    mx = lon
    my = log(tan((lat + tau/4)/2))
    res = 2**(zoom + ZOOM_OFFSET) / tau
    px = mx*res
    py = my*res
    return px, py

def pixels2latlon(px, py, zoom):
    res = 2**(zoom + ZOOM_OFFSET) / tau
    mx = px/res
    my = py/res
    lon = mx
    lat = 2*atan(exp(my)) - tau/4
    return lat, lon

# Google Maps API requires URLs to be signed with you
# encoded API secret.
def sign_url(input_url=None, secret=None):
    """ Sign a request URL with a URL signing secret.

      signed_url = sign_url(input_url=my_url, secret=SECRET)

      Args:
      input_url - The URL to sign
      secret    - Your URL signing secret

      Returns:
      The signed request URL
    """

    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")

    url = urlparse.urlparse(input_url)

    # We only need to sign the path+query part of the string
    url_to_sign = url.path + "?" + url.query

    # Decode the private key into its binary format
    # We need to decode the URL-encoded private key
    decoded_key = base64.urlsafe_b64decode(secret)

    # Create a signature using the private key and the URL-encoded
    # string using HMAC SHA1. This signature will be binary.
    signature = hmac.new(decoded_key, url_to_sign.encode('utf-8'), hashlib.sha1)

    # Encode the binary signature into base64 for use within a URL
    encoded_signature = base64.urlsafe_b64encode(signature.digest())

    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Return signed URL
    #print(encoded_signature)
    #print(str(encoded_signature))
    return original_url + "&signature=" + encoded_signature.decode('utf-8')

#Stitch together Google Maps images from lat, long coordinates
#Based on work by heltonbiker and BenElgar
#Changes:
#* updated for Python 3
#* added Google Maps API key (compliance with T&C, although can set to None)
#* handle http request exceptions

#With contributions from Eric Toombs.
#Changes:
#* Dramatically simplified the maths.
#* Set a more reasonable default logo cutoff.
#* Added global constants for logo cutoff and max image size.
#* Translated a couple presumably Portuguese variable names to English.
def get_maps_image(NW_lat_long, SE_lat_long, zoom=18,maptype='satellite',scale=1):
    '''
    Downloads and stitches together static Google Maps images for
    a given rectangular region.

    Args:
    NW_lat_long,SE_lat_long: Coordinates for upper left and lower right
                            corners of the region of interest
    zoom: Level of detail within maps - higher is more detailed, and will
            result in more images being downloaded
    maptype: 'satellite' or 'roadmap'
    scale: Affects number of pixels returned (i.e. scale=2 returns 2x pixels
            within the same region)
    '''

    ullat, ullon = NW_lat_long
    lrlat, lrlon = SE_lat_long

    # convert all these coordinates to pixels
    ulx, uly = latlon2pixels(ullat, ullon, zoom)
    lrx, lry = latlon2pixels(lrlat, lrlon, zoom)

    # calculate total pixel dimensions of final image
    dx, dy = lrx - ulx, uly - lry
    # calculate rows and columns
    cols, rows = ceil(dx/MAXSIZE), ceil(dy/MAXSIZE)

    # calculate pixel dimensions of each small image
    width = ceil(dx/cols)
    height = ceil(dy/rows)
    heightplus = height + LOGO_CUTOFF

    # assemble the image from stitched
    final = Image.new('RGB', (int(dx), int(dy)))
    print('Downloading',str(rows*cols),'images.')
    for x in range(cols):
        for y in range(rows):
            dxn = width * (0.5 + x)
            dyn = height * (0.5 + y)
            latn, lonn = pixels2latlon(
                    ulx + dxn, uly - dyn - LOGO_CUTOFF/2, zoom)
            position = ','.join((str(latn/DEGREE), str(lonn/DEGREE)))
            #print(x, y, position)
            urlparams = {
                    'center': position,
                    'zoom': str(zoom),
                    'size': '%dx%d' % (width, heightplus),
                    'maptype': maptype,
                    #'sensor': 'false',
                    'scale': scale
                }
            if GOOGLE_MAPS_API_KEY is not None:
                urlparams['key'] = GOOGLE_MAPS_API_KEY

            url=GOOGLE_API_URL
            for key,val in urlparams.items():
                url=url+key+'='+str(val)+'&'
            url=url[:-1]
            url=sign_url(url,GOOGLE_MAPS_SECRET)
            try:
                response = requests.get(url)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(e)
                sys.exit(1)

            im = Image.open(BytesIO(response.content))
            final.paste(im, (int(x*width), int(y*height)))
            im.close()
    return final



# Grabs run data from Strava API
from stravalib.client import Client

client = Client()
authorize_url = client.authorization_url(client_id=STRAVA_CLIENT_ID, redirect_uri='http://localhost:8080')
client = Client(access_token=STRAVA_ACCESS_TOKEN)
runs=list(client.get_activities(limit=100))
runs=runs[::-1]

# Gets Google Map data for region
fig = plt.figure(figsize=(16, 24))
subplt=plt.subplot(111)
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.97,top=0.97)

chicago_lat=41.9418065
chicago_lon=-87.636318
m = Basemap(projection='lcc', resolution='h',
            width=basemap_width, height=basemap_height,
            lat_0=chicago_lat, lon_0=chicago_lon,ax=subplt,fix_aspect=False)

w,n=m(xmin,ymax,inverse=True)
e,s=m(xmax,ymin,inverse=True)

# circumference/radius
tau = 2*np.pi
DEGREE = tau/360
NW_lat_long =  (n*DEGREE, w*DEGREE)
SE_lat_long = (s*DEGREE, e*DEGREE)

# 16 is sufficient for city scale images
# 18 is good for ~few block regions
zoom = 18   # be careful not to get too many images!

bkgd_filename='chi_satellite.png'
result = get_maps_image(NW_lat_long, SE_lat_long, zoom=zoom,maptype='satellite',scale=1)
result.save(bkgd_filename)


# Animates all runs simultaneously
desired_fps=60
length_in_sec=30
time_delay=0.5 # Delay in seconds before launching the animation for next run
frame_delay=int(time_delay*desired_fps)
frames=int(desired_fps)*int(length_in_sec)

# Initialize the matplotlib plot with the Basemap region
# of Chicago
print('Initializing... ')
fig = plt.figure(figsize=(24, 36))
subplt=plt.subplot(111)
fig.subplots_adjust(left=0.03,bottom=0.03,right=0.97,top=0.97)
m = Basemap(projection='lcc', resolution='h',
            width=basemap_width, height=basemap_height,
            lat_0=chicago_lat, lon_0=chicago_lon,ax=subplt,fix_aspect=False)

run_name='Chicago Marathon'
run=[i for i in runs if i.name==run_name][0]

plot_data=[]
r=client.get_activity_streams(run.id,types=['latlng'],resolution='medium')
if r is not None:
    lats,lngs=zip(*r['latlng'].data)
    x, y = m(lngs, lats)
    plot_data={'name':run.name,'x':x,'y':y}


# Interpolate the X/Y data to achieve the desired number of points
# necessary for framerate
xdata=np.interp(np.linspace(0,999,frames),*list(zip(*enumerate(plot_data['x']))))
ydata=np.interp(np.linspace(0,999,frames),*list(zip(*enumerate(plot_data['y']))))


frame_size=750
subplt.set_xlim(xdata[0]-frame_size,xdata[0]+frame_size)
subplt.set_ylim(ydata[0]-frame_size,ydata[0]+frame_size)
subplt.get_xaxis().set_ticks([])
subplt.get_yaxis().set_ticks([])

with writer.saving(fig, 'output/'+run.name+'.mp4', 100):
    writer.grab_frame()
    line=matplotlib.lines.Line2D([],[],linestyle='-',linewidth=15,alpha=0.6,color=col)
    header=matplotlib.lines.Line2D([],[],marker='o',markersize=40,color=col,linestyle='None')
    subplt.add_line(line)
    subplt.add_line(header)

    start=time.time()
    smoothing=15
    for i in range(1,len(xdata)):
        line.set_xdata(xdata[:i])
        line.set_ydata(ydata[:i])
        header.set_xdata(xdata[i])
        header.set_ydata(ydata[i])
        new_cenx=np.mean(xdata[i-smoothing:i])
        new_ceny=np.mean(ydata[i-smoothing:i])
        if i>=smoothing:
            subplt.set_xlim(new_cenx-frame_size,new_cenx+frame_size)
            subplt.set_ylim(new_ceny-frame_size,new_ceny+frame_size)

        plt.draw()
        writer.grab_frame()
print('Done.')
