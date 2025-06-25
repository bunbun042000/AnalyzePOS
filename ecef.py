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
    fact, dddmm = np.modf(dmsdeg/100.0)
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
        self.data['x'] = __x
        self.data['y'] = __y
        self.data['z'] = __z

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
        self.data = pd.DataFrame(data=[[__x, __y, __z]], columns=['x', 'y', 'z'])
        return self

    def Setblhdeg(self, __lat, __long, __height):
        v = ecef.a / math.sqrt(1 - ecef.e2 * (math.sin(math.radians(__lat))) ** 2)
        temp_x = [(v + __height) * math.cos(math.radians(__lat)) * math.cos(
            math.radians(__long))]
        temp_y = [(v + __height) * math.cos(math.radians(__lat)) * math.sin(
            math.radians(__long))]
        temp_z = [(v * (1 - ecef.e2) + __height) * math.sin(math.radians(__lat))]
        self.data = pd.DataFrame({'x':temp_x, 'y':temp_y, 'z':temp_z})
        return self

    def Setblhdeg_array(self, pos_in_degree):
        v = (ecef.a / np.sqrt(1 - ecef.e2 * (np.sin(np.radians(pos_in_degree['latitude'].to_numpy()))) ** 2))
        temp_x = ((v + pos_in_degree['height'].to_numpy()) * np.cos(np.radians(pos_in_degree['latitude'].to_numpy())) * np.cos(
           np.radians(pos_in_degree['longitude'].to_numpy())))
        temp_y = ((v + pos_in_degree['height'].to_numpy()) * np.cos(np.radians(pos_in_degree['latitude'].to_numpy())) * np.sin(
           np.radians(pos_in_degree['longitude'].to_numpy())))
        temp_z = ((v * (1 - ecef.e2) + pos_in_degree['height'].to_numpy()) * np.sin(np.radians(pos_in_degree['latitude'].to_numpy())))
        self.data = pd.DataFrame({'x':temp_x, 'y':temp_y, 'z':temp_z})
        return self

    def Setblhdms(self, pos):
        pos['latitude'] = dms2deg(pos['latitude'])
        pos['longitude'] = dms2deg(pos['longitude'])
        return self.Setblhdeg_array(pos)

    def Getxyz(self):
        return self.data

    def Getblhdeg(self):
        longitudes = np.degrees(np.arctan2(self.data['y'], self.data['x']))
        p = np.sqrt(self.data['x'] ** 2 + self.data['y'] ** 2)
        r = np.sqrt(p ** 2 + self.data['z'] ** 2)
        u = np.arctan2(self.data['z'] * ((1 - ecef.f) + ecef.e2 * ecef.a / r), p)
        latitudes = np.degrees(np.arctan2(self.data['z'] * (1 - ecef.f) + ecef.e2 * ecef.a * (np.sin(u) ** 3),
                                          (1 - ecef.f) * (p - ecef.e2 * ecef.a * (np.cos(u)) ** 3)))
        heights = p * np.cos(np.radians(latitudes)) + self.data['z'] * np.sin(np.radians(latitudes)) - ecef.a * np.sqrt(1 - ecef.e2 * (np.sin(np.radians(latitudes))) ** 2)

        ret_val = pd.DataFrame()
        ret_val['latitude'] = latitudes
        ret_val['longitude'] = longitudes
        ret_val['height'] = heights

        return ret_val

    def Getblhdms(self):
        tempblh = self.Getblhdeg()
        tempblh['latitude'] = deg2dms(tempblh['latitude'])
        tempblh['longitude'] = deg2dms(tempblh['longitude'])
        return tempblh

class enu():
    position = pd.DataFrame()

    def __init__(self, __positions=None, __origin=None):
        origin = __origin
        origin_blhdeg = origin.Getblhdeg()

        vec = __positions.Getxyz().to_numpy() - __origin.Getxyz().iloc[0].to_numpy()
        rot = np.array([[-math.sin(math.radians(origin_blhdeg.at[0, 'longitude'])), math.cos(math.radians(origin_blhdeg.at[0, 'longitude'])), 0],
                                   [-math.cos(math.radians(origin_blhdeg.at[0, 'longitude'])) * math.sin(math.radians(origin_blhdeg.at[0, 'latitude'])),
                                    -math.sin(math.radians(origin_blhdeg.at[0, 'longitude'])) * math.sin(math.radians(origin_blhdeg.at[0, 'latitude'])),
                                    math.cos(math.radians(origin_blhdeg.at[0, 'latitude']))],
                                   [math.cos(math.radians(origin_blhdeg.at[0, 'longitude'])) * math.cos(math.radians(origin_blhdeg.at[0, 'latitude'])),
                                   math.sin(math.radians(origin_blhdeg.at[0, 'longitude'])) * math.cos(math.radians(origin_blhdeg.at[0, 'latitude'])),
                                    math.sin(math.radians(origin_blhdeg.at[0, 'latitude']))]])

        enu = rot @ vec.T
        self.position = pd.DataFrame(data=enu.T, columns=['e', 'n', 'u'])
        # print(self.position)
        self.position['origin_longitude'] = origin_blhdeg['longitude']
        self.position['origin_latitude'] = origin_blhdeg['latitude']
        self.position['origin_height'] = origin_blhdeg['height']
        std = self.position.std()
        self.position['2drms'] = 2 * math.sqrt(std.at['e'] ** 2 + std.at['n'] ** 2)

    def GetENU(self):
        return self.position

    def SetDate(self, date):
        self.position['GPST'] = pd.to_datetime(date, format='%Y-%m-%d %x')

    def SetNsat(self, nsat):
        self.position['ns'] = nsat

    def SetQ(self, Q):
        self.position['Q'] = Q
