import sys
import os
import pandas as pd
import datetime
import dateutil.parser
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.dates import DateFormatter

sys.path.append('.')
import ecef


def read_posfile(__processing_filename):
    with open(__processing_filename) as _f:
        _i = 0
        for _s_line in _f:
            if "%  GPST" in _s_line:
                break
            _i += 2

    _result = pd.read_csv(__processing_filename, skiprows=_i,  
                          low_memory=False, skipinitialspace=True, encoding='utf-8',
                          on_bad_lines='skip', sep='\\s+',
                          names=('GPS_date',  'GPS_time', 'latitude', 'longitude',
                 'height', 'Q', 'ns', 'sdn(m)', 'sde(m)', 'sdu(m)', 'sdne(m)', 'sdeu(m)',
                   'sdun(m)', 'age(s)', 'ratio'))

    _positions = _result[['latitude', 'longitude', 'height', 'Q', 'ns']].astype(
        {'latitude': float, 'longitude': float, 'height': float})
    _positions['GPST'] = pd.to_datetime(_result['GPS_date'].str.cat(_result['GPS_time'], sep=' '), unit='ns')
    return _positions

def convert_enu(__positions, __quality, __origin):
    positions = __positions
    ratio_of_quality = positions['Q'].value_counts(dropna=False, normalize=True)
    pos_in_xyz = ecef.ecef().Setblhdeg_array(positions)

    if __origin == None:
        if __quality == 99:
            pos_mean = ecef.ecef().Setblhdeg_array().Getxyz().mean()
        else:
            pos_mean = ecef.ecef().Setblhdeg_array(positions[positions['Q'] == __quality]).Getxyz().mean()

        print('x = ', pos_mean['x'], ' y = ', pos_mean['y'], ' z = ', pos_mean['z'])
        position_origin = ecef.ecef().Setxyz(pos_mean['x'], pos_mean['y'], pos_mean['z'])
        print('Latitiude = ', position_origin.Getblhdeg().at[0, 'latitude'], '  Longitude = ', position_origin.Getblhdeg().at[0, 'longitude'], ' height = ', position_origin.Getblhdeg().at[0, 'height']);
    else:
        position_origin = __origin

    enu = ecef.enu(pos_in_xyz, position_origin)
    enu.SetDate(positions['GPST'])
    enu.SetQ(positions['Q'])
    enu.SetNsat(positions['ns'])
    print("e = ", enu.GetENU()['e'].mean(), " n = ", enu.GetENU()['n'].mean(), " u = ", enu.GetENU()['u'].mean(), "\n")
    print("Quality ratio = \n", ratio_of_quality)
    print("2Drms = ", enu.GetENU().at[0, '2drms'])

    return enu, position_origin

def color(value):
    if value == 1:
        return 'b'
    else:
        return 'r'

def plot_route(__enu, figname=None, labelname=None):
    enu = __enu

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.015
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)

    en_x = enu.GetENU()['e'].to_numpy()
    en_y = enu.GetENU()['n'].to_numpy()
    en_x_q1 = enu.GetENU()[enu.GetENU()['Q'] == 1]['e'].to_numpy()
    en_y_q1 = enu.GetENU()[enu.GetENU()['Q'] == 1]['n'].to_numpy()
    en_x_q2 = enu.GetENU()[enu.GetENU()['Q'] == 2]['e'].to_numpy()
    en_y_q2 = enu.GetENU()[enu.GetENU()['Q'] == 2]['n'].to_numpy()
#    colormap = plt.get_cmap('tab10')
#    col = colormap(enu.GetENU()['Q'].to_numpy())

    ax.scatter(en_x_q1,en_y_q1, c='g', s=1)
    ax.scatter(en_x_q2,en_y_q2, c='orange', s=1)
#    ax.scatter(en_x, en_y, c=enu.GetENU()['Q'], alpha=1, cmap=cm.rainbow)
        
    ax.legend(["test"])
    ax.grid(True)
    ax.set_aspect('equal')

    binwidth = 0.1
    xymax = max(np.max(np.abs(en_x)), np.max(np.abs(en_y)))
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)

    fig.patch.set_alpha(0)
    if (figname != ""):
        plt.savefig(figname)

    plt.show()

def plot_scatter(__enu1, __enu2, q, figname=None):
    enu1 = __enu1
    enu2 = __enu2

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.015
    range = 0.01
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)
    ax.set_aspect('equal')

#    ax_histx = fig.add_axes(rect_histx, sharex=ax)
#    ax_histy = fig.add_axes(rect_histy, sharey=ax)

#    ax_histx.tick_params(axis="x", labelbottom=False)
#    ax_histy.tick_params(axis="y", labelleft=False)

    en1_x_temp = enu1.GetENU()[(np.abs(enu1.GetENU()['e']) < range)]['e']
    en1_y_temp = enu1.GetENU()[(np.abs(enu1.GetENU()['e']) < range)]['n']
    en1_x = en1_x_temp[np.abs(en1_y_temp) < range].to_numpy()
    en1_y = en1_y_temp[np.abs(en1_y_temp) < range].to_numpy()

    if en1_x.size != 0:
        ax.scatter(en1_x, en1_y, c="b", alpha=1, label='Kinematic')

    en2_x = enu2.GetENU()['e'].to_numpy()
    en2_y = enu2.GetENU()['n'].to_numpy()
    if en2_x.size != 0:
        ax.scatter(en2_x, en2_y, c="r", alpha=1, label='Static')
        two_drms_en2 = 2 * np.sqrt(en2_x.std() ** 2 + en2_y.std() ** 2)
        print("2Drms of 2 = " + str(two_drms_en2) + "\n")
        if q == 99:
            circle2 = plt.Circle((enu2.GetENU()['e'].mean(), enu2.GetENU()['n'].mean()), two_drms_en2, fill=False)
        else:
            circle2 = plt.Circle((enu2.GetENU()[enu2.GetENU()['Q'] == q]['e'].mean(), enu2.GetENU()[enu2.GetENU()['Q'] == q]['n'].mean()), two_drms_en2, fill=False)
        ax.add_patch(circle2)

    ax.legend()
    ax.grid(True)

    binwidth = 0.0001
    xymax = range
    lim = (int(xymax / binwidth) + 1) * binwidth

#    bins = np.arange(-lim, lim + binwidth, binwidth)
#    ax_histx.hist([en1_x, en2_x], bins=bins)
    #    ax_histx.hist(enu2.GetENU()['e'].to_numpy(), bins=bins, color="r")
#    ax_histy.hist([en1_y, en2_y], bins=bins, orientation="horizontal")

    fig.patch.set_alpha(0)
    if (figname != ""):
        plt.savefig(figname)

    plt.show()

def plot_position(__enu1, __enu2, figname=None):
    enu1 = __enu1
    enu2 = __enu2

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)

    e1 = pd.Series(enu1.GetENU()['e'].to_numpy(), index=enu1.GetENU()['GPST'].to_numpy()).dropna()
    n1 = pd.Series(enu1.GetENU()['n'].to_numpy(), index=enu1.GetENU()['GPST'].to_numpy()).dropna()
    u1 = pd.Series(enu1.GetENU()['u'].to_numpy(), index=enu1.GetENU()['GPST'].to_numpy()).dropna()

    e2 = pd.Series(enu2.GetENU()['e'].to_numpy(), index=enu2.GetENU()['GPST'].to_numpy()).dropna()
    n2 = pd.Series(enu2.GetENU()['n'].to_numpy(), index=enu2.GetENU()['GPST'].to_numpy()).dropna()
    u2 = pd.Series(enu2.GetENU()['u'].to_numpy(), index=enu2.GetENU()['GPST'].to_numpy()).dropna()

    start = datetime.datetime(2023, 6, 22, 22, 57, 13)
    end = datetime.datetime(2021, 6, 22, 23, 30, 15)
    axes[0].plot(e1[start:end], color='b')
    axes[0].plot(e2[start:end], color='r')
    axes[0].set_ylabel('e')
    axes[1].plot(n1[start:end], color='b')
    axes[1].plot(n2[start:end], color='r')
    axes[1].set_ylabel('n')
    axes[2].plot(u1[start:end], color='b')
    axes[2].plot(u2[start:end], color='r')
    axes[2].set_ylabel('u')

    if (figname != ""):
        fig.savefig(figname)

    fig.show()

def plot_visible_satellites(__enu1, figname=None):
    enu1 = __enu1

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,3))

    ns1 = pd.Series(enu1.GetENU()[enu1.GetENU()['Q'] == 1]['ns'].to_numpy(), index=enu1.GetENU()[enu1.GetENU()['Q'] == 1]['GPST'].to_numpy()).dropna()
    ns2 = pd.Series(enu1.GetENU()[enu1.GetENU()['Q'] == 2]['ns'].to_numpy(), index=enu1.GetENU()[enu1.GetENU()['Q'] == 2]['GPST'].to_numpy()).dropna()
    axes.set_ylabel('ns')
    axes.set_xlabel('time')
    axes.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    axes.bar(ns1.index, ns1.to_numpy(), color='g')
    axes.bar(ns2.index, ns2.to_numpy(), color='orange')

    if (figname != ""):
        fig.savefig(figname)

    fig.show()

def main():
    args = sys.argv

    np.set_printoptions(precision=10)
    pd.options.display.float_format = '{:.6f}'.format
    #mode = "Single"
    mode1 = "Kinematic"
    quality = 1
    mode2 = "Static"
    #mode = "Static"

    basedir = "H:\\GNSS_data\\Solutions\\19K004_with_19K003_Kinematic\\"
    origin = ecef.ecef()
    roverstation = "19K004"
    # Station code: TR36444421702 別当前
    # Latitude: 43 00 32.2380 Longitude: 144 20 30.2284 Height: 6.56 Geoid Height 35.93 (=02P203)
#    origin.Setxyz(-3795474.870, 2723135.845, 4328258.266)
#    baseStation1 = "02P203" # kushiro
    #origin.Setxyz(-3798939.894, 2722643.392, 4325545.429)
    # baseStation1 = "950124" # Akan
    #baseStation1 = "05S052" # Shoro
    #origin.Setxyz(-3787385.589, 2738175.488, 4325882.996)
    #baseStation2 = "02P203" # kushiro
    baseStation = "19K003"
    origin.Setblhdeg(43.0157662, 144.26100490, 49.312)
    #Origin.Setxyz(-3791223.295156889, 2728184.801227155, 4328820.62403418)
#    frequency1 = "L1"
#    frequency2 = "L1+L2"
    startdate = "20241212"
    enddate = "20241212"
    time1 = "095033"
    time2 = "101628"
    elevationMask1 = "ElevMask_20"
    duration = pd.date_range(startdate, enddate)
    for i in duration:
        date = i.strftime("%Y%m%d")
        print(date)

        processing_filename1 = basedir + roverstation + "_with_" + baseStation + "_BS_" + mode1 + "_" + date + "-" + time1 + "_LatLong" + ".pos"
#        processing_filename2 = basedir + roverstation + "_" + mode2 + "_with_" + baseStation + "\\" + roverstation + "_" + date + "_" + time2 + ".pos"
        positions1 = read_posfile(processing_filename1)
#        positions2 = read_posfile(processing_filename2)

#        enu2, orig2 = convert_enu(positions2, quality, origin)
        enu1, orig1 = convert_enu(positions1, quality, origin)

#        save_figname = basedir + roverstation + "_" + mode1 + "_with_" + baseStation + "_vs_with_" + mode2 + "_with_" + baseStation + "_" + date + "_" + time2 + ".pdf"
#        plot_scatter(enu1, enu2, quality, save_figname)
        save_figname2 = basedir + roverstation + "_" + mode1 + "_with_" + baseStation + "_" + date + "_" + time2 + "_route.pdf"
        plot_route(enu1, save_figname2)
        save_figname3 = basedir + roverstation + "_" + mode1 + "_with_" + baseStation + "_" + date + "_" + time2 + "_ns.pdf"
        plot_visible_satellites(enu1, save_figname3)

#        output_filename1 = basedir + roverstation + "_" + mode1 + "_with_" + baseStation + "_" + date + "_" + time + "_ENU_.csv"
#        output_filename2 = basedir + roverstation + "_" + mode2 + "_with_" + baseStation + "_" + date + "_" + time + "_ENU_.csv"

#        enu1.GetENU().to_csv(output_filename1, index=False, columns=['GPST', 'e', 'n', 'u', 'Q', 'ns', 'origin_latitude', 'origin_longitude', 'origin_height', '2drms'])
#        enu2.GetENU().to_csv(output_filename2, index=False, columns=['GPST', 'e', 'n', 'u', 'Q', 'ns', 'origin_latitude', 'origin_longitude', 'origin_height', '2drms'])

if __name__ == '__main__':
    main()
