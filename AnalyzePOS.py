import sys
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
                         skiprows=_i, parse_dates=[0, 1], low_memory=False, skipinitialspace=True).rename(
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

    enu1 = ecef.enu(pos_in_xyz, position_origin)
    enu1.SetDate(positions['GPST'])
    enu1.SetQ(positions['Q'])
    enu1.SetNsat(positions['ns'])
    print("e = ", enu1.GetENU()['e'].mean(), " n = ", enu1.GetENU()['n'].mean(), " u = ", enu1.GetENU()['u'].mean(), "\n")
    print("Quality ratio = \n", ratio_of_quality)
    print("2Drms of 1 = ", enu1.GetENU().at[0, '2drms'])

    return enu1, position_origin

def plot_scatter(__enu1, __enu2, q, figname=None):
    enu1 = __enu1
    enu2 = __enu2

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
    ax.scatter(en1_x, en1_y, c="b", alpha=1, label='with 02P203')
    en2_x = enu2.GetENU()[enu2.GetENU()['Q'] == q]['e'].to_numpy()
    en2_y = enu2.GetENU()[enu2.GetENU()['Q'] == q]['n'].to_numpy()
    ax.scatter(en2_x, en2_y, c="r", alpha=1, label='with 19K003')
    ax.legend()
    ax.grid(True)
    two_drms_en1 = 2 * np.sqrt(en1_x.std() ** 2 + en1_y.std() ** 2)
    two_drms_en2 = 2 * np.sqrt(en2_x.std() ** 2 + en2_y.std() ** 2)
    print("2Drms of 1 = " + str(two_drms_en1) + "\n")
    print("2Drms of 2 = " + str(two_drms_en2) + "\n")
    circle1 = plt.Circle((enu1.GetENU()[enu1.GetENU()['Q'] == q]['e'].mean(), enu1.GetENU()[enu1.GetENU()['Q'] == q]['n'].mean()), two_drms_en1, fill=False)
    circle2 = plt.Circle((enu2.GetENU()[enu2.GetENU()['Q'] == q]['e'].mean(), enu2.GetENU()[enu2.GetENU()['Q'] == q]['n'].mean()), two_drms_en2, fill=False)

    ax.add_patch(circle1)
    ax.add_patch(circle2)

    binwidth = 0.0001
    xymax = max(np.max(np.abs(en1_x)), np.max(np.abs(en1_y)))
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist([en1_x, en2_x], bins=bins)
    #    ax_histx.hist(enu2.GetENU()['e'].to_numpy(), bins=bins, color="r")
    ax_histy.hist([en1_y, en2_y], bins=bins, orientation="horizontal")

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

    start = datetime.datetime(2020, 12, 16, 0, 0, 0)
    end = datetime.datetime(2020, 12, 17, 0, 0, 0)
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

def plot_visible_satellites(__enu1, __enu2, figname=None):
    enu1 = __enu1
    enu2 = __enu2

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,3))

    ns1 = pd.Series(enu1.GetENU()['ns'].to_numpy(), index=enu1.GetENU()['GPST'].to_numpy()).dropna()
    ns2 = pd.Series(enu2.GetENU()['ns'].to_numpy(), index=enu2.GetENU()['GPST'].to_numpy()).dropna()
    axes.set_ylabel('ns')
    axes.set_xlabel('time')
    axes.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    axes.plot(ns1[datetime.datetime(2020, 12, 16, 0, 0, 0):], color='b')
    axes.plot(ns2[datetime.datetime(2020, 12, 17, 0, 0, 0):], color='r')

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

    basedir = "C:\\home\\bun\\PyCharmProjects\\AnalyzePOS\\"
    origin = ecef.ecef()
    roverstation = "19K004"
    baseStation1 = "02P203" # kushiro
    origin.Setxyz(-3798939.894, 2722643.392, 4325545.429)
    # baseStation1 = "950124" # Akan
    #baseStation1 = "05S052" # Shoro
    #origin.Setxyz(-3787385.589, 2738175.488, 4325882.996)
    #baseStation2 = "02P203" # kushiro
    baseStation2 = "19K003"
    frequency1 = "L1"
    frequency2 = "L1+L2"
    startdate = "20201216"
    enddate = "20201216"
#    time = "065732"
    time = "000000"
    satellites = "G,J"
    duration = pd.date_range(startdate, enddate)
    for i in duration:
        date = i.strftime("%Y%m%d")
        print(date)

        processing_filename1 = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "\\" + roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "_" + date + "_" + time + "_" + satellites + ".pos"
        processing_filename2 = basedir + roverstation + "_" + mode + "_with_" + baseStation2 + "_freq_" + frequency2 + "\\" + roverstation + "_" + mode + "_with_" + baseStation2 + "_freq_" + frequency2 + "_" + date + "_" + time + "_" + satellites + ".pos"
        positions1 = read_posfile(processing_filename1)
        positions2 = read_posfile(processing_filename2)

        enu2, orig2 = convert_enu(positions2, quality, None)
        enu1, orig1 = convert_enu(positions1, quality, orig2)

        save_figname = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "_vs_with_" + baseStation2 + "_freq_" + frequency2 + "_with_Q=" + str(quality) + "_" + date + "_" + time + ".pdf"
        plot_scatter(enu1, enu2, quality, save_figname)
        save_figname2 = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "_vs_with_" + baseStation2 + "_freq_" + frequency2 + "_with_Q=" + str(quality) + "_" + date + "_" + time + "_position.pdf"
        plot_position(enu1, enu2, save_figname2)
        save_figname3 = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "_vs_with_" + baseStation2 + "_freq_" + frequency2 + "_with_Q=" + str(quality) + "_" + date + "_" + time + "_ns.pdf"
        plot_visible_satellites(enu1, enu2, save_figname3)

        output_filename1 = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "_" + date + "_" + time + "_" + satellites + "_ENU_.csv"
        output_filename2 = basedir + roverstation + "_" + mode + "_with_" + baseStation2 + "_freq_" + frequency2 + "_" + date + "_" + time + "_" + satellites + "_ENU_.csv"

        enu1.GetENU().to_csv(output_filename1, index=False, columns=['GPST', 'e', 'n', 'u', 'Q', 'ns', 'origin_latitude', 'origin_longitude', 'origin_height', '2drms'])
        enu2.GetENU().to_csv(output_filename2, index=False, columns=['GPST', 'e', 'n', 'u', 'Q', 'ns', 'origin_latitude', 'origin_longitude', 'origin_height', '2drms'])

if __name__ == '__main__':
    main()
