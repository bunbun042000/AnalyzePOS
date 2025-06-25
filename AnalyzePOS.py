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
                         skiprows=_i, parse_dates=[0, 1], low_memory=False, skipinitialspace=True).rename(
        columns={'%  GPST                ': 'GPST', 'latitude(deg)': 'latitude', 'longitude(deg)': 'longitude',
                 'height(m)': 'height'})

    _positions = _result[['GPST', 'latitude', 'longitude', 'height', 'Q', 'ns']].astype(
        {'latitude': float, 'longitude': float, 'height': float})
    return _positions

def convert_enu(__positions, __quality, __origin):
    positions = __positions
    ratio_of_quality = positions['Q'].value_counts(dropna=False, normalize=True)
    temp_ecef = ecef.ecef()
    pos_in_xyz = temp_ecef.Setblhdeg_array(positions)

#    print("pos_in_xyz\n", pos_in_xyz)

    if __origin == None:
        pos_mean = ecef.ecef().Setblhdeg_array(positions).Getxyz().mean()
#        print('x = ', pos_mean['x'], ' y = ', pos_mean['y'], ' z = ', pos_mean['z'])
        position_origin = ecef.ecef(pos_mean['x'], pos_mean['y'], pos_mean['z'])
#        print('Latitiude = ', position_origin.Getblhdeg().at[0, 'latitude'], '  Longitude = ', position_origin.Getblhdeg().at[0, 'longitude'], ' height = ', position_origin.Getblhdeg().at[0, 'height']);
    else:
        position_origin = __origin

#    print(position_origin)
    enu = ecef.enu(pos_in_xyz, position_origin)
    enu.SetDate(positions['GPST'])
#    print(enu)
    enu.SetQ(positions['Q'])
    enu.SetNsat(positions['ns'])
#    print("e = ", enu.GetENU().loc[:, 'e'].mean(), " n = ", enu.GetENU().loc[:, 'n'].mean(), " u = ", enu.GetENU().loc[:, 'u'].mean(), "\n")
    print("Quality ratio = \n", ratio_of_quality)
#    print("2Drms = ", enu.GetENU()['2drms'])

    return enu, position_origin

def plot_track(__enu1, q, figname=None):
    enu1 = __enu1

    en1_x = enu1.GetENU().loc[:, 'e'].to_numpy().tolist()
    en1_y = enu1.GetENU().loc[:, 'n'].to_numpy().tolist()

    print(en1_x)

    plt.figure(figsize=(8,8))
    plt.scatter(en1_x, en1_y, s=1, marker=".")

    g = plt.subplot()
    g.set_ylim([-1000,5000])
    g.set_xlim([0,6000])


    if (figname != ""):
        plt.savefig(figname)

    plt.show()

def plot_tracks(__enu1, __enu2, __enu3, q, figname=None):
    enu1 = __enu1
    enu2 = __enu2
    enu3 = __enu3

    en1_x = enu1.GetENU().loc[:, 'e'][enu1.GetENU()['Q'] == q].to_numpy()
    en1_y = enu1.GetENU().loc[:, 'n'][enu1.GetENU()['Q'] == q].to_numpy()
    en2_x = enu2.GetENU().loc[:, 'e'].to_numpy()
    en2_y = enu2.GetENU().loc[:, 'n'].to_numpy()
    en3_x = enu3.GetENU().loc[:, 'e'][enu3.GetENU()['Q'] == q].to_numpy()
    en3_y = enu3.GetENU().loc[:, 'n'][enu3.GetENU()['Q'] == q].to_numpy()

#    plt.figure(figsize=(8,8))
#    plt.scatter(en1_x, en1_y, c="red", s=10, marker=".")
#    plt.scatter(en2_x, en2_y, c="green", s=10, marker=".")
#    plt.scatter(en3_x, en3_y, c="blue", s=10, marker=".")

    g, axes = plt.subplots()
    axes.set_aspect('equal')

    if en1_x.size != 0:
        axes.scatter(en1_x, en1_y, c="b", alpha=1, label='with 19K003')
        two_drms_en1 = 2 * np.sqrt(en1_x.std() ** 2 + en1_y.std() ** 2)
        print("2Drms of 1 = " + str(two_drms_en1) + "\n")
        circle1 = plt.Circle((enu1.GetENU()[enu1.GetENU()['Q'] == q]['e'].mean(), enu1.GetENU()[enu1.GetENU()['Q'] == q]['n'].mean()), two_drms_en1, fill=False)
        axes.add_patch(circle1)

    if en2_x.size != 0:
        axes.scatter(en2_x, en2_y, c="g", alpha=1, label='with 19K003')
        two_drms_en2 = 2 * np.sqrt(en2_x.std() ** 2 + en2_y.std() ** 2)
        print("2Drms of 2 = " + str(two_drms_en2) + "\n")
        circle2 = plt.Circle((enu2.GetENU()['e'].mean(), enu2.GetENU()['n'].mean()), two_drms_en2, fill=False)
        axes.add_patch(circle2)

    if en3_x.size != 0:
        axes.scatter(en3_x, en3_y, c="r", alpha=1, label='with 19K003')
        two_drms_en3 = 2 * np.sqrt(en3_x.std() ** 2 + en3_y.std() ** 2)
        print("2Drms of 3 = " + str(two_drms_en3) + "\n")
        circle3 = plt.Circle((enu3.GetENU()[enu3.GetENU()['Q'] == q]['e'].mean(), enu3.GetENU()[enu3.GetENU()['Q'] == q]['n'].mean()), two_drms_en3, fill=False)
        axes.add_patch(circle3)
    axes.set_ylim([-0.5,4])
    axes.set_xlim([-0.5,4])


    if (figname != ""):
        plt.savefig(figname)

    plt.show()

def plot_scatters(__enu1, __enu2, __enu3, q, figname=None):
    enu1 = __enu1
    enu2 = __enu2
    enu3 = __enu3

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

    en1_x = enu1.GetENU().loc[:, 'e'][enu1.GetENU()['Q'] == q].to_numpy()
    en1_y = enu1.GetENU().loc[:, 'n'][enu1.GetENU()['Q'] == q].to_numpy()
    en2_x = enu2.GetENU().loc[:, 'e'][enu2.GetENU()['Q'] == q].to_numpy()
    en2_y = enu2.GetENU().loc[:, 'n'][enu2.GetENU()['Q'] == q].to_numpy()
    en3_x = enu3.GetENU().loc[:, 'e'][enu3.GetENU()['Q'] == q].to_numpy()
    en3_y = enu3.GetENU().loc[:, 'n'][enu3.GetENU()['Q'] == q].to_numpy()

    if en1_x.size != 0:
        ax.scatter(en1_x, en1_y, c="b", alpha=1, label='with 19K003')
        two_drms_en1 = 2 * np.sqrt(en1_x.std() ** 2 + en1_y.std() ** 2)
        print("2Drms of 1 = " + str(two_drms_en1) + "\n")
        circle1 = plt.Circle((enu1.GetENU()[enu1.GetENU()['Q'] == q]['e'].mean(), enu1.GetENU()[enu1.GetENU()['Q'] == q]['n'].mean()), two_drms_en1, fill=False)
        ax.add_patch(circle1)

    if en2_x.size != 0:
        ax.scatter(en2_x, en2_y, c="g", alpha=1, label='with 19K003')
        two_drms_en2 = 2 * np.sqrt(en2_x.std() ** 2 + en2_y.std() ** 2)
        print("2Drms of 1 = " + str(two_drms_en2) + "\n")
        circle2 = plt.Circle((enu2.GetENU()[enu2.GetENU()['Q'] == q]['e'].mean(), enu2.GetENU()[enu2.GetENU()['Q'] == q]['n'].mean()), two_drms_en2, fill=False)
        ax.add_patch(circle2)

    if en3_x.size != 0:
        ax.scatter(en3_x, en3_y, c="r", alpha=1, label='with 19K003')
        two_drms_en3 = 2 * np.sqrt(en3_x.std() ** 2 + en3_y.std() ** 2)
        print("2Drms of 1 = " + str(two_drms_en3) + "\n")
        circl31 = plt.Circle((enu3.GetENU()[enu3.GetENU()['Q'] == q]['e'].mean(), enu3.GetENU()[enu3.GetENU()['Q'] == q]['n'].mean()), two_drms_en3, fill=False)
        ax.add_patch(circle3)

    ax.legend()
    ax.grid(True)

    binwidth = 0.0001
    xymax = np.max([np.abs(en1_x), np.abs(en2_x), np.abs(en3_x), np.abs(en1_y), np.abs(en2_y), np.abs(en3_y)])
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(en1_x, bins=bins)
    #    ax_histx.hist(enu2.GetENU()['e'].to_numpy(), bins=bins, color="r")
    ax_histy.hist(en1_y, bins=bins, orientation="horizontal")

    fig.patch.set_alpha(0)
    if (figname != ""):
        plt.savefig(figname)

    plt.show()


def plot_scatter(__enu1, q, figname=None):
    enu1 = __enu1

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
        ax.scatter(en1_x, en1_y, c="b", alpha=1, label='with 19K003')
        two_drms_en1 = 2 * np.sqrt(en1_x.std() ** 2 + en1_y.std() ** 2)
        print("2Drms of 1 = " + str(two_drms_en1) + "\n")
        circle1 = plt.Circle((enu1.GetENU()[enu1.GetENU()['Q'] == q]['e'].mean(), enu1.GetENU()[enu1.GetENU()['Q'] == q]['n'].mean()), two_drms_en1, fill=False)
        ax.add_patch(circle1)


    ax.legend()
    ax.grid(True)

    binwidth = 0.0001
    xymax = max(np.max(np.abs(en1_x)), np.max(np.abs(en1_y)))
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(en1_x, bins=bins)
    #    ax_histx.hist(enu2.GetENU()['e'].to_numpy(), bins=bins, color="r")
    ax_histy.hist(en1_y, bins=bins, orientation="horizontal")

    fig.patch.set_alpha(0)
    if (figname != ""):
        plt.savefig(figname)

    plt.show()

def plot_position(__enu1, figname=None):
    enu1 = __enu1

    fig, axes = plt.subplots(3, 1, sharex=True)

#    print(enu1.GetENU()['GPST'].to_numpy())

    e1 = pd.Series(enu1.GetENU().loc[:, 'e'].to_numpy().tolist(), index=enu1.GetENU().loc[:, 'GPST']).dropna()
    n1 = pd.Series(enu1.GetENU().loc[:, 'n'].to_numpy().tolist(), index=enu1.GetENU().loc[:, 'GPST']).dropna()
    u1 = pd.Series(enu1.GetENU().loc[:, 'u'].to_numpy().tolist(), index=enu1.GetENU().loc[:, 'GPST']).dropna()

    start = datetime.datetime(2022, 6, 17, 4, 53, 00)
    end = datetime.datetime(2022, 6, 17, 5, 30, 0)
    axes[0].plot(e1[start:end], color='b')
    axes[0].set_ylabel('e')
    axes[1].plot(n1[start:end], color='b')
    axes[1].set_ylabel('n')
    axes[2].plot(u1[start:end], color='b')
    axes[2].set_ylabel('u')
    axes[2].set_ylim(bottom=-20,top=30)

    if (figname != ""):
        fig.savefig(figname)

    fig.show()

def plot_positions(__enu1, __enu2, __enu3, figname=None):
    enu1 = __enu1
    enu2 = __enu2
    enu3 = __enu3

    fig, axes = plt.subplots(3, 1, sharex=True)

#    print(enu1.GetENU()['GPST'].to_numpy())

    e1 = pd.Series(enu1.GetENU().loc[:, 'e'].to_numpy().tolist(), index=enu1.GetENU().loc[:, 'GPST']).dropna()
    n1 = pd.Series(enu1.GetENU().loc[:, 'n'].to_numpy().tolist(), index=enu1.GetENU().loc[:, 'GPST']).dropna()
    u1 = pd.Series(enu1.GetENU().loc[:, 'u'].to_numpy().tolist(), index=enu1.GetENU().loc[:, 'GPST']).dropna()

    e2 = pd.Series(enu2.GetENU().loc[:, 'e'].to_numpy().tolist(), index=enu2.GetENU().loc[:, 'GPST']).dropna()
    n2 = pd.Series(enu2.GetENU().loc[:, 'n'].to_numpy().tolist(), index=enu2.GetENU().loc[:, 'GPST']).dropna()
    u2 = pd.Series(enu2.GetENU().loc[:, 'u'].to_numpy().tolist(), index=enu2.GetENU().loc[:, 'GPST']).dropna()

    e3 = pd.Series(enu3.GetENU().loc[:, 'e'].to_numpy().tolist(), index=enu3.GetENU().loc[:, 'GPST']).dropna()
    n3 = pd.Series(enu3.GetENU().loc[:, 'n'].to_numpy().tolist(), index=enu3.GetENU().loc[:, 'GPST']).dropna()
    u3 = pd.Series(enu3.GetENU().loc[:, 'u'].to_numpy().tolist(), index=enu3.GetENU().loc[:, 'GPST']).dropna()

    start = datetime.datetime(2022, 6, 17, 4, 53, 00)
    end = datetime.datetime(2022, 6, 17, 5, 30, 0)
    axes[0].plot(e1[start:end], color='b')
    axes[1].plot(n1[start:end], color='b')
    axes[2].plot(u1[start:end], color='b')

    axes[0].plot(e2[start:end], color='g')
    axes[1].plot(n2[start:end], color='g')
    axes[2].plot(u2[start:end], color='g')

    axes[0].plot(e3[start:end], color='r')
    axes[1].plot(n3[start:end], color='r')
    axes[2].plot(u3[start:end], color='r')

    axes[2].set_ylim(bottom=-20,top=30)

    axes[0].set_ylabel('e')
    axes[1].set_ylabel('n')
    axes[2].set_ylabel('u')

    if (figname != ""):
        fig.savefig(figname)

    fig.show()

def plot_visible_satellites(__enu1, figname=None):
    enu1 = __enu1

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,3))

    ns1 = pd.Series(enu1.GetENU()['ns'].to_numpy(), index=enu1.GetENU()['GPST'].to_numpy()).dropna()
    axes.set_ylabel('ns')
    axes.set_xlabel('time')
    axes.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    axes.set_ylim(bottom=0,top=max(ns1)+1)
    axes.plot(ns1[datetime.datetime(2022, 6, 17, 4, 53, 00):], color='b')

    if (figname != ""):
        fig.savefig(figname)

    fig.show()

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

    basedir = ""
    figdir="..\\Figs\\"
    roverstation = "22K006"
    # Station code: TR36444421702 別当前
    # Latitude: 43 00 32.2380 Longitude: 144 20 30.2284 Height: 6.56 Geoid Height 35.93 (=02P203)
    origin = ecef.ecef(-3791219.47403825, 2728182.06088074, 4328816.25046466)
#    print(origin.__str__())
    baseStation1 = "19K003"
    #baseStation1 = "02P203" # kushiro
    #origin.Setxyz(-3798939.894, 2722643.392, 4325545.429)
    # baseStation1 = "950124" # Akan
    #baseStation1 = "05S052" # Shoro
    #origin.Setxyz(-3787385.589, 2738175.488, 4325882.996)
    #baseStation2 = "02P203" # kushiro
    #Origin.Setxyz(-3791223.295156889, 2728184.801227155, 4328820.62403418)
    origin2 = ecef.ecef().Setblhdeg(43.014738371, 144.262577202, 52.7012)
#    print(origin2.__str__())

#    frequency1 = "L1"
    frequency2 = "L1+L2"
    startdate = "20220617"
    enddate = "20220617"
    time = "044500"
#    time = "000000"
    time2 = "045300"
    satellites = "G,R,E,J,C,I"
    duration = pd.date_range(startdate, enddate)
    for i in duration:
        date = i.strftime("%Y%m%d")
        print(date)

        processing_filename1 = roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "_" + date + "_" + time + "_" + satellites + ".pos"
        positions1 = read_posfile(processing_filename1)

        enu1, orig1 = convert_enu(positions1, quality, origin)

        processing_filename2 = roverstation + "_Static_with_" + baseStation1 + "_freq_" + frequency2 + "_" + date + "_" + time2 + "_" + satellites + ".pos"
        positions2 = read_posfile(processing_filename2)


        processing_filename3 = roverstation + "_Single_freq_" + frequency2 + "_" + date + "_" + time2 + "_" + satellites + ".pos"
        positions3 = read_posfile(processing_filename3)

        processing_filename4 = roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "_" + date + "_" + time2 + "_" + satellites + ".pos"
        positions4 = read_posfile(processing_filename4)

        enu2_with_Origin2, orig2 = convert_enu(positions2, quality, None)
        enu3_with_Origin2, orig3 = convert_enu(positions3, quality, orig2)
        enu4_with_Origin2, orig3 = convert_enu(positions4, quality, orig2)

        processing_filename5 = roverstation + "_Single_freq_" + frequency2 + "_" + date + "_" + time + "_" + satellites + ".pos"
        positions5 = read_posfile(processing_filename5)

        enu5, orig3 = convert_enu(positions5, quality, origin)

        save_figname = figdir + roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "_with_Q=" + str(quality) + "_" + date + "_" + time + ".pdf"
        plot_track(enu1, quality, save_figname)
        save_figname5 = figdir + roverstation + "_Single_freq_" + frequency2 + "_with_Q=" + str(quality) + "_" + date + "_" + time + ".pdf"
        plot_track(enu5, quality, save_figname5)
        save_figname4 = figdir + roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "_with_Q=" + str(quality) + "_" + date + "_" + time + "_vs_Static_vs_Single.pdf"
        plot_tracks(enu4_with_Origin2, enu3_with_Origin2, enu2_with_Origin2, quality, save_figname4)
        save_figname2 = figdir + roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "_with_Q=" + str(quality) + "_" + date + "_" + time + "_position.pdf"
        plot_positions(enu4_with_Origin2, enu3_with_Origin2, enu2_with_Origin2, save_figname2)
        save_figname3 = figdir + roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "_with_Q=" + str(quality) + "_" + date + "_" + time + "_ns.pdf"
        plot_visible_satellites(enu1, save_figname3)

        output_filename1 = basedir + roverstation + "_" + mode + "_with_" + baseStation1 + "_freq_" + frequency2 + "_" + date + "_" + time + "_" + satellites + "_ENU_.csv"

        enu1.GetENU().to_csv(output_filename1, index=False, columns=['GPST', 'e', 'n', 'u', 'Q', 'ns', 'origin_latitude', 'origin_longitude', 'origin_height', '2drms'])

if __name__ == '__main__':
    main()
