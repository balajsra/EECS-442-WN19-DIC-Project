#############################################
#       EECS 442: Computer Vision - W19     #
#############################################
# Authors: Sravan Balaji & Kevin Monpara    #
# Filename: main.py                         #
# Description:                              #
#############################################

import numpy as np


class FrameData:
    load = 0    # Load in N
    disp = 0    # Displacement in mm

    def __init__(self, load, disp):
        self.load = load
        self.disp = disp


def calc_stress():
    file = open("../Section001_Data.txt", "r")

    w = 0   # Specimen Width in mm
    t = 0   # Specimen Thickness in mm
    gl = 0  # Specimen Gauge Length in mm
    ol = 0  # Specimen Overall Length in mm

    load_disp_data = dict()

    data_start = -1
    in_frame_data = False

    for line in file:
        if "Width" in line:
            index = line.find("\t")
            w = float(line[index + 1:])

        if "Thickness" in line:
            index = line.find("\t")
            t = float(line[index + 1:])

        if "Gauge Length" in line:
            index = line.find("\t")
            gl = float(line[index + 1:])

        if "Overall Length" in line:
            index = line.find("\t")
            ol = float(line[index + 1:])

        if "Frame #" in line:
            data_start = 2

        if data_start != -1:
            data_start -= 1

        if data_start == 0:
            in_frame_data = True

        if in_frame_data:
            if line != '\t\t\n':
                frame_num_str, load_str, disp_str = line.split("\t")
                frame_num = int(frame_num_str)
                load = float(load_str)
                disp = float(disp_str)
                load_disp_data[frame_num] = FrameData(load, disp)

    print("Done")


if __name__ == "__main__":
    calc_stress()
