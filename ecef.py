import os
import subprocess
import shutil
import sys
import datetime
import dateutil.parser
import numpy as np
import math
import pandas as pd


def dms2deg(dmsdeg):
    fact, dddmmm = np.modf(dmsdeg/100.0)
    mm, ddd = np.modf(dddmm/100.0)
    deg = ddd + 10.0 / 6.0 * mm + fact/36.0
    return deg

def deg2dms(deg):
    fact, ddd = np.modf(deg)
    fact2, mm = np.modf(fact * 60)
    dmsdeg = (ddd * 100 + mm) * 100 + fact2 * 60
    return dmsdeg

class GPSDate(datetime.datetime):
    def __new__(cls, __date):
        self = super().__new__(cls, __date.year, __date.month, __date.day, __date.hour, __date.minute, __date.second,
                               __date.microsecond, __date.tzinfo)
        delta = self - datetime.datetime(1980, 1, 6, 0, 0)
        self.weeks = delta.days // 7
        self.days = delta.days % 7
        return self

    def gpsWeek(self):
        return str(self.weeks).zfill(4)

    def gpsDay(self):
        return str(self.days).zfill(1)

class ecef(pd.DataFrame):
    a = 6378137.0  # meter
    f = 1.0 / 298.257223563  # WGS84
    e2 = f * (2 - f)

    data = pd.DataFrame()

    def __init__(self, __x=None, __y=None, __z=None):
        if (__x is not None) and (__y is not None) and (__z is not None):
#            print("x = ", __x, " y = ", __y, " z = ", __z)
            self.data = pd.DataFrame(data = {'x': __x, 'y': __y, 'z': __z}, index={'1st'})

    def __eq__(self, other):
        if other == None:
            return False
        else:
            return (self.data['x'] == other.data['x']) & (self.data['y'] == other.data['y']) & (self.data['z'] == other.data['z'])

    def __str__(self):
        return "x = " + str(self.data['x']) + "\ny = " + str(self.data['y']) + "\nz = " + str(self.data['z']) + "\n"

    def Setxyz_array(self, pos_in_xyz):
        self.data['x'] = pos_in_xyz['x']
        self.data['y'] = pos_in_xyz['y']
        self.data['z'] = pos_in_xyz['z']
        return self

    def Setxyz(self, __x, __y, __z):
        self.data = pd.DataFrame(data={'x': __x, 'y': __y, 'z': __z})
        return self

    def Setblhdeg(self, __lat, __long, __height):
        v = (ecef.a / np.sqrt(1 - ecef.e2 * (np.sin(np.radians(__lat))) ** 2))
        temp_x = ((v + __height) * np.cos(np.radians(__lat)) * np.cos(np.radians(__long)))
        temp_y = ((v + __height) * np.cos(np.radians(__lat)) * np.sin(np.radians(__long)))
        temp_z = ((v * (1 - ecef.e2) + __height) * np.sin(np.radians(__lat)))
        self.data = pd.DataFrame(data = {'x': temp_x, 'y': temp_y, 'z': temp_z}, index={'1st'})
        return self

    def Setblhdeg_array(self, pos_in_degree):
        lats = np.radians(pos_in_degree['latitude'])
        longs = np.radians(pos_in_degree['longitude'])
        heights = pos_in_degree['height']

        v = (ecef.a / np.sqrt(1 - ecef.e2 * (np.sin(lats)) ** 2))
        temp_x = ((v + heights) * np.cos(lats) * np.cos(longs))
        temp_y = ((v + heights) * np.cos(lats) * np.sin(longs))
        temp_z = ((v * (1 - ecef.e2) + heights) * np.sin(lats))
        self.data = pd.DataFrame(data ={'x': temp_x, 'y': temp_y, 'z': temp_z})
        return self

    def Setblhdms(self, pos):
        pos['latitude'] = dms2deg(pos['latitude'])
        pos['longitude'] = dms2deg(pos['longitude'])
        return self.Setblhdeg_array(pos)

    def Getxyz(self):
        return pd.DataFrame(data = {'x': self.data['x'], 'y': self.data['y'], 'z': self.data['z']})

    def Getblhdeg(self):
#       print("arctan2 = " + str(np.rad2deg(np.arctan2(self.data.at[0, 'y'], self.data.at[0, 'x']))) + "\n")
        longitudes = np.rad2deg(np.arctan2(self.data['y'], self.data['x']))
#        print(longitudes)
        p = np.sqrt(self.data['x'] ** 2 + self.data['y'] ** 2)
        r = np.sqrt(p ** 2 + self.data['z'] ** 2)
        u = np.arctan2(self.data['z'] * ((1 - ecef.f) + ecef.e2 * ecef.a / r), p)
        latitudes = np.degrees(np.arctan2(self.data['z'] * (1 - ecef.f) + ecef.e2 * ecef.a * (np.sin(u) ** 3),
                                          (1 - ecef.f) * (p - ecef.e2 * ecef.a * (np.cos(u)) ** 3)))
        heights = p * np.cos(np.radians(latitudes)) + self.data['z'] * np.sin(np.radians(latitudes)) - ecef.a * np.sqrt(1 - ecef.e2 * (np.sin(np.radians(latitudes))) ** 2)

 #       print("lat = " + str(latitudes) + " long = " + str(longitudes) + " height = " + str(heights) + "\n")

        ret_val = pd.DataFrame(data = {'latitude': latitudes, 'longitude':longitudes, 'height': heights})

        return ret_val

    def Getblhdms(self):
        tempblh = self.Getblhdeg()
        tempblh['latitude'] = deg2dms(tempblh['latitude'])
        tempblh['longitude'] = deg2dms(tempblh['longitude'])
        return tempblh

    def GetDate(self):
        return self.data['GPST']

class enu():
    position = pd.DataFrame()

    def __init__(self, __positions=None, __origin=None):
        origin = __origin
        origin_blhdeg = origin.Getblhdeg()


#        print(__positions.Getxyz().to_numpy())
#        print(origin.Getxyz().loc['1st', :].to_numpy())
        vec = __positions.Getxyz().to_numpy() - origin.Getxyz().loc['1st', :].to_numpy()
#        print(vec)

        rot = np.array([[-np.sin(np.radians(origin_blhdeg.at['1st', 'longitude'])),
                         np.cos(np.radians(origin_blhdeg.at['1st', 'longitude'])), 
                         0],
                        [-np.cos(np.radians(origin_blhdeg.at['1st', 'longitude'])) * np.sin(np.radians(origin_blhdeg.at['1st', 'latitude'])),
                        -np.sin(np.radians(origin_blhdeg.at['1st', 'longitude'])) * np.sin(np.radians(origin_blhdeg.at['1st', 'latitude'])),
                         np.cos(np.radians(origin_blhdeg.at['1st', 'latitude']))],
                        [np.cos(np.radians(origin_blhdeg.at['1st', 'longitude'])) * np.cos(np.radians(origin_blhdeg.at['1st', 'latitude'])), 
                        np.sin(np.radians(origin_blhdeg.at['1st', 'longitude'])) * np.cos(np.radians(origin_blhdeg.at['1st', 'latitude'])),
                         np.sin(np.radians(origin_blhdeg.at['1st', 'latitude']))]])

#        print(rot)
        enu = (rot @ (vec.T)).T
        self.position = pd.DataFrame(data = {'e': enu[:, 0], 'n': enu[:, 1], 'u': enu[:, 2]})
#        print(self.position)
        self.position['origin_longitude'] = origin_blhdeg.at['1st', 'longitude']
        self.position['origin_latitude'] = origin_blhdeg.at['1st', 'latitude']
        self.position['origin_height'] = origin_blhdeg.at['1st', 'height']
        pos_std = np.std(self.position)
 #       print(pos_std)
        self.position['2drms'] =  2 * np.sqrt((pos_std[0]) ** 2 + (pos_std[1]) ** 2)

    def GetENU(self):
        return self.position

    def SetDate(self, date):
        self.position['GPST'] = pd.to_datetime(date)
#        print("In SetDate\n", self.position)

    def SetNsat(self, nsat):
        self.position['ns'] = nsat

    def SetQ(self, Q):
        self.position['Q'] = Q


