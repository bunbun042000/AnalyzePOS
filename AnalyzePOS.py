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
            _i += 1

    _result = pd.read_csv(__processing_filename,
                         sep=' ', skiprows=_i, parse_dates=[0, 1], low_memory=False, skipinitialspace=True, encoding='utf-8').rename(
        columns={'%  GPST                ': 'GPST', 'latitude(deg)': 'latitude', 'longitude(deg)': 'longitude',
                 'height(m)': 'height'})

    _positions = _result[['GPST', 'latitude', 'longitude', 'height', 'Q', 'ns']].astype(
        {'latitude': float, 'longitude': float, 'height': float})
    return _positions

def convert_enu(__positions, __quality, __origin):
    positions = __positions
    ratio_of_quality = positions['Q'].value_counts(dropna=False, normalize=True)
    pos_in_xyz = ecef.ecef().Setblhdeg_array(positions)

    if __origin == None:
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

def plot_scatter(__enu1, __enu2, __enu3, __enu4, __enu5, q, figname=None):
    enu1 = __enu1
    enu2 = __enu2
    enu3 = __enu3
    enu4 = __enu4
    enu5 = __enu5

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.015
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)
    ax.set_aspect('equal')

    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    en1_x = enu1.GetENU()[enu1.GetENU()['Q'] == q]['e'].to_numpy()
    en1_y = enu1.GetENU()[enu1.GetENU()['Q'] == q]['n'].to_numpy()
    if en1_x.size != 0:
        ax.scatter(en1_x, en1_y, c="b", alpha=1, label='without SNR mask')
        two_drms_en1 = 2 * np.sqrt(en1_x.std() ** 2 + en1_y.std() ** 2)
        print("2Drms of 1 = " + str(two_drms_en1) + "\n")
        print("mean : e = " + str(enu1.GetENU()[enu1.GetENU()['Q'] == q]['e'].mean()) + "n = " + str(enu1.GetENU()[enu1.GetENU()['Q'] == q]['n'].mean()) + "\n")
        circle1 = plt.Circle((enu1.GetENU()[enu1.GetENU()['Q'] == q]['e'].mean(), enu1.GetENU()[enu1.GetENU()['Q'] == q]['n'].mean()), two_drms_en1, fill=False)
        ax.add_patch(circle1)

    en2_x = enu2.GetENU()[enu2.GetENU()['Q'] == q]['e'].to_numpy()
    en2_y = enu2.GetENU()[enu2.GetENU()['Q'] == q]['n'].to_numpy()
    if en2_x.size != 0:
        ax.scatter(en2_x, en2_y, c="r", alpha=1, label='with SNR mask and Elevation mask')
        two_drms_en2 = 2 * np.sqrt(en2_x.std() ** 2 + en2_y.std() ** 2)
        print("2Drms of 2 = " + str(two_drms_en2) + "\n")
        print("mean : e = " + str(enu2.GetENU()[enu2.GetENU()['Q'] == q]['e'].mean()) + "n = " + str(enu2.GetENU()[enu2.GetENU()['Q'] == q]['n'].mean()) + "\n")
        circle2 = plt.Circle((enu2.GetENU()[enu2.GetENU()['Q'] == q]['e'].mean(), enu2.GetENU()[enu2.GetENU()['Q'] == q]['n'].mean()), two_drms_en2, fill=False)
        ax.add_patch(circle2)

    en3_x = enu3.GetENU()[enu3.GetENU()['Q'] == q]['e'].to_numpy()
    en3_y = enu3.GetENU()[enu3.GetENU()['Q'] == q]['n'].to_numpy()
    if en3_x.size != 0:
        ax.scatter(en3_x, en3_y, c="g", alpha=1, label='with SNR mask and Elevation mask 20')
        two_drms_en3 = 2 * np.sqrt(en3_x.std() ** 2 + en3_y.std() ** 2)
        print("2Drms of 3 = " + str(two_drms_en3) + "\n")
        print("mean : e = " + str(enu3.GetENU()[enu3.GetENU()['Q'] == q]['e'].mean()) + "n = " + str(enu3.GetENU()[enu3.GetENU()['Q'] == q]['n'].mean()) + "\n")
        circle3 = plt.Circle((enu3.GetENU()[enu3.GetENU()['Q'] == q]['e'].mean(), enu3.GetENU()[enu3.GetENU()['Q'] == q]['n'].mean()), two_drms_en3, fill=False)
        ax.add_patch(circle3)

    en4_x = enu4.GetENU()[enu4.GetENU()['Q'] == q]['e'].to_numpy()
    en4_y = enu4.GetENU()[enu4.GetENU()['Q'] == q]['n'].to_numpy()
    if en4_x.size != 0:
        ax.scatter(en4_x, en4_y, c="c", alpha=1, label='with SNR mask and Elevation mask 30')
        two_drms_en4 = 2 * np.sqrt(en4_x.std() ** 2 + en4_y.std() ** 2)
        print("2Drms of 4 = " + str(two_drms_en4) + "\n")
        print("mean : e = " + str(enu4.GetENU()[enu4.GetENU()['Q'] == q]['e'].mean()) + "n = " + str(enu4.GetENU()[enu4.GetENU()['Q'] == q]['n'].mean()) + "\n")
        circle4 = plt.Circle((enu4.GetENU()[enu4.GetENU()['Q'] == q]['e'].mean(), enu4.GetENU()[enu4.GetENU()['Q'] == q]['n'].mean()), two_drms_en4, fill=False)
        ax.add_patch(circle4)

    en5_x = enu5.GetENU()[enu5.GetENU()['Q'] == q]['e'].to_numpy()
    en5_y = enu5.GetENU()[enu5.GetENU()['Q'] == q]['n'].to_numpy()
    if en5_x.size != 0:
        ax.scatter(en5_x, en5_y, c="y", alpha=1, label='with SNR mask and Elevation mask 40')
        two_drms_en5 = 2 * np.sqrt(en5_x.std() ** 2 + en5_y.std() ** 2)
        print("2Drms of 5 = " + str(two_drms_en5) + "\n")
        print("mean : e = " + str(enu5.GetENU()[enu5.GetENU()['Q'] == q]['e'].mean()) + "n = " + str(enu5.GetENU()[enu5.GetENU()['Q'] == q]['n'].mean()) + "\n")
        circle5 = plt.Circle((enu5.GetENU()[enu5.GetENU()['Q'] == q]['e'].mean(), enu5.GetENU()[enu5.GetENU()['Q'] == q]['n'].mean()), two_drms_en5, fill=False)
        ax.add_patch(circle5)

    ax.legend()
    ax.grid(True)

    binwidth = 0.0001
    np.amax(np.abs(en1_x))
    np.amax(np.abs(en2_x))
    xymax = np.amax([np.amax(np.abs(en1_x)), np.amax(np.abs(en2_x)), np.amax(np.abs(en1_y)), np.amax(np.abs(en2_y))])
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist([en1_x, en2_x, en3_x, en4_x, en5_x], bins=bins)
    #    ax_histx.hist(enu2.GetENU()['e'].to_numpy(), bins=bins, color="r")
    ax_histy.hist([en1_y, en2_y, en3_y, en4_y, en5_y], bins=bins, orientation="horizontal")

    fig.patch.set_alpha(0)
    if (figname != ""):
        plt.savefig(figname)

    plt.show()

def plot_position(__enu1, __enu2, __enu3, __enu4, __enu5, q, figname=None):
    enu1 = __enu1
    enu2 = __enu2
    enu3 = __enu3
    enu4 = __enu4
    enu5 = __enu5

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)

    e1 = pd.Series(enu1.GetENU()[enu1.GetENU()['Q'] == q]['e'].to_numpy(), index=enu1.GetENU()[enu1.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()
    n1 = pd.Series(enu1.GetENU()[enu1.GetENU()['Q'] == q]['n'].to_numpy(), index=enu1.GetENU()[enu1.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()
#    u1 = pd.Series(enu1.GetENU()[enu1.GetENU()['Q'] == q]['u'].to_numpy(), index=enu1.GetENU()[enu1.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()

    e2 = pd.Series(enu2.GetENU()[enu2.GetENU()['Q'] == q]['e'].to_numpy(), index=enu2.GetENU()[enu2.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()
    n2 = pd.Series(enu2.GetENU()[enu2.GetENU()['Q'] == q]['n'].to_numpy(), index=enu2.GetENU()[enu2.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()
#    u2 = pd.Series(enu2.GetENU()[enu2.GetENU()['Q'] == q]['u'].to_numpy(), index=enu2.GetENU()[enu2.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()

    e3 = pd.Series(enu3.GetENU()[enu3.GetENU()['Q'] == q]['e'].to_numpy(), index=enu3.GetENU()[enu3.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()
    n3 = pd.Series(enu3.GetENU()[enu3.GetENU()['Q'] == q]['n'].to_numpy(), index=enu3.GetENU()[enu3.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()
#    u3 = pd.Series(enu3.GetENU()[enu3.GetENU()['Q'] == q]['u'].to_numpy(), index=enu3.GetENU()[enu3.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()

    e4 = pd.Series(enu4.GetENU()[enu4.GetENU()['Q'] == q]['e'].to_numpy(), index=enu4.GetENU()[enu4.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()
    n4 = pd.Series(enu4.GetENU()[enu4.GetENU()['Q'] == q]['n'].to_numpy(), index=enu4.GetENU()[enu4.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()
#    u4 = pd.Series(enu4.GetENU()[enu4.GetENU()['Q'] == q]['u'].to_numpy(), index=enu4.GetENU()[enu4.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()

    e5 = pd.Series(enu5.GetENU()[enu5.GetENU()['Q'] == q]['e'].to_numpy(), index=enu5.GetENU()[enu5.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()
    n5 = pd.Series(enu5.GetENU()[enu5.GetENU()['Q'] == q]['n'].to_numpy(), index=enu5.GetENU()[enu5.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()
#    u5 = pd.Series(enu5.GetENU()[enu5.GetENU()['Q'] == q]['u'].to_numpy(), index=enu5.GetENU()[enu5.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()

    start = datetime.datetime(2022, 12, 16, 6, 52, 39)
    end = datetime.datetime(2022, 12, 16, 7, 0, 0)
    axes[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
#    axes[0].set_xlim(start,end)
    axes[0].plot(e1, color='b')
    axes[0].plot(e2, color='r')
    axes[0].plot(e3, color='g')
    axes[0].plot(e4, color='c')
    axes[0].plot(e5, color='y')
    axes[0].set_ylabel('e')
    axes[1].plot(n1, color='b')
    axes[1].plot(n2, color='r')
    axes[1].plot(n3, color='g')
    axes[1].plot(n4, color='c')
    axes[1].plot(n5, color='y')
    axes[1].set_ylabel('n')
#    axes[2].plot(u1, color='b')
#    axes[2].plot(u2, color='r')
#    axes[2].plot(u3, color='g')
#    axes[2].plot(u4, color='c')
#    axes[2].plot(u5, color='y')
#    axes[2].set_ylabel('u')

    if (figname != ""):
        fig.savefig(figname)

    fig.show()

def plot_updown(__enu1, __enu2, __enu3, __enu4, __enu5, q, figname=None):
    enu1 = __enu1
    enu2 = __enu2
    enu3 = __enu3
    enu4 = __enu4
    enu5 = __enu5

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,3))

    u1 = pd.Series(enu1.GetENU()[enu1.GetENU()['Q'] == q]['u'].to_numpy(), index=enu1.GetENU()[enu1.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()

    u2 = pd.Series(enu2.GetENU()[enu2.GetENU()['Q'] == q]['u'].to_numpy(), index=enu2.GetENU()[enu2.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()

    u3 = pd.Series(enu3.GetENU()[enu3.GetENU()['Q'] == q]['u'].to_numpy(), index=enu3.GetENU()[enu3.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()

    u4 = pd.Series(enu4.GetENU()[enu4.GetENU()['Q'] == q]['u'].to_numpy(), index=enu4.GetENU()[enu4.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()

    u5 = pd.Series(enu5.GetENU()[enu5.GetENU()['Q'] == q]['u'].to_numpy(), index=enu5.GetENU()[enu5.GetENU()['Q'] == q]['GPST'].to_numpy()).dropna()

    start = datetime.datetime(2022, 12, 16, 6, 52, 39)
    end = datetime.datetime(2022, 12, 16, 7, 0, 0)
    axes.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#    axes[0].set_xlim(start,end)
    axes.plot(u1, color='b')
    axes.plot(u2, color='r')
    axes.plot(u3, color='g')
    axes.plot(u4, color='c')
    axes.plot(u5, color='y')
    axes.set_ylabel('u')

    if (figname != ""):
        fig.savefig(figname)

    fig.show()

def plot_visible_satellites(__enu1, __enu2, __enu3, __enu4, __enu5, figname=None):
    enu1 = __enu1
    enu2 = __enu2
    enu3 = __enu3
    enu4 = __enu4
    enu5 = __enu5

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,3))

    ns1 = pd.Series(enu1.GetENU()['ns'].to_numpy(), index=enu1.GetENU()['GPST'].to_numpy()).dropna()
    ns2 = pd.Series(enu2.GetENU()['ns'].to_numpy(), index=enu2.GetENU()['GPST'].to_numpy()).dropna()
    ns3 = pd.Series(enu3.GetENU()['ns'].to_numpy(), index=enu3.GetENU()['GPST'].to_numpy()).dropna()
    ns4 = pd.Series(enu4.GetENU()['ns'].to_numpy(), index=enu4.GetENU()['GPST'].to_numpy()).dropna()
    ns5 = pd.Series(enu5.GetENU()['ns'].to_numpy(), index=enu5.GetENU()['GPST'].to_numpy()).dropna()
    axes.set_ylabel('ns')
    axes.set_xlabel('time')
    axes.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    start = datetime.datetime(2022, 12, 16, 6, 52, 39)
    end = datetime.datetime(2022, 12, 16, 7, 0, 0)
#    axes.set_xlim(start,end)
    axes.plot(ns1, color='b')
    axes.plot(ns2, color='r')
    axes.plot(ns3, color='g')
    axes.plot(ns4, color='c')
    axes.plot(ns5, color='y')

    if (figname != ""):
        fig.savefig(figname)

    fig.show()

def main():
    args = sys.argv

    np.set_printoptions(precision=10)
    pd.options.display.float_format = '{:.6f}'.format
    #mode = "Single"
    mode = "Kinematic"
    #mode = "Static"
    if mode == "Single":
        quality = 5
    elif mode == "Kinematic":
        quality = 1
    elif mode == "Static":
        quality = 1

    basedir = os.getcwd() + "\\..\\..\\..\\GNSS_data\\Solution\\"
    print(basedir)
    origin = ecef.ecef()
    roverstation = "19K004"
    # Station code: TR36444421702 別当前
    # Latitude: 43 00 32.2380 Longitude: 144 20 30.2284 Height: 6.56 Geoid Height 29.8170 
    # https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh/calc_f.html
    origin.Setblhdeg(43.008955, 144.341730111, 36.377)
    print(origin.Getxyz())
    baseStation1 = "19K003"
    #baseStation1 = "02P203" # kushiro
    #origin.Setxyz(-3798939.894, 2722643.392, 4325545.429)
    # baseStation1 = "950124" # Akan
    #baseStation1 = "05S052" # Shoro
    #origin.Setxyz(-3787385.589, 2738175.488, 4325882.996)
    #baseStation2 = "02P203" # kushiro
    #Origin.Setxyz(-3791223.295156889, 2728184.801227155, 4328820.62403418)
#    frequency1 = "L1"
    frequency2 = "L1+L2"
    startdate = "20221216"
    enddate = "20221216"
    time = "065239"
#    time = "000000"
    satellites = "G,J"
    duration = pd.date_range(startdate, enddate)
    for i in duration:
        date = i.strftime("%Y%m%d")
        print(date)

        processing_filename1 = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "\\K0043500_ElevationMask=0_SNRMask=modified3.pos"
        print(processing_filename1)
        positions1 = read_posfile(processing_filename1)
        enu1, orig1 = convert_enu(positions1, quality, origin)

        processing_filename2 = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "\\K0043500_ElevationMask=10.pos"
        positions1 = read_posfile(processing_filename2)
        enu2, orig2 = convert_enu(positions1, quality, origin)

        processing_filename3 = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "\\K0043500_ElevationMask=20.pos"
        positions1 = read_posfile(processing_filename3)
        enu3, orig3 = convert_enu(positions1, quality, origin)

        processing_filename4 = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "\\K0043500_ElevationMask=30.pos"
        positions1 = read_posfile(processing_filename4)
        enu4, orig4 = convert_enu(positions1, quality, origin)

        processing_filename5 = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "\\K0043500_ElevationMask=40.pos"
        positions1 = read_posfile(processing_filename5)
        enu5, orig5 = convert_enu(positions1, quality, origin)

        save_figname = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "_with_Q=" + str(quality) + "_" + date + "_" + time + "_0_40.pdf"
        plot_scatter(enu1, enu2, enu3, enu4, enu5, quality, save_figname)
        save_figname2 = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "_with_Q=" + str(quality) + "_" + date + "_" + time + "_0_40_position.pdf"
        plot_position(enu1, enu2, enu3, enu4, enu5, quality, save_figname2)
        save_figname3 = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "_with_Q=" + str(quality) + "_" + date + "_" + time + "_0_40_updown.pdf"
        plot_updown(enu1, enu2, enu3, enu4, enu5, quality, save_figname3)
        save_figname4 = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "_with_Q=" + str(quality) + "_" + date + "_" + time + "_0_40_ns.pdf"
        plot_visible_satellites(enu1, enu2, enu3, enu4, enu5, save_figname4)

#        output_filename1 = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "_" + date + "_" + time + "_" + satellites + "_withoutElevMask_and_SNRMask_ENU_.csv"
#        output_filename2 = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "_" + date + "_" + time + "_" + satellites + "_withElevMask_and_SNRMask_ENU_.csv"

#        enu1.GetENU().to_csv(output_filename1, index=False, columns=['GPST', 'e', 'n', 'u', 'Q', 'ns', 'origin_latitude', 'origin_longitude', 'origin_height', '2drms'])
#        enu2.GetENU().to_csv(output_filename2, index=False, columns=['GPST', 'e', 'n', 'u', 'Q', 'ns', 'origin_latitude', 'origin_longitude', 'origin_height', '2drms'])

if __name__ == '__main__':
    main()
