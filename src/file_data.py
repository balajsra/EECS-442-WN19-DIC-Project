#############################################################
#             EECS 442: Computer Vision - W19               #
#############################################################
# Authors: Sravan Balaji & Kevin Monpara                    #
# Filename: file_data.py                                    #
# Description:                                              #
#   Read data file to get specimen dimensions.              #
#   Read in load and displacement data for each frame.      #
#   Calculate stress from load and cross-sectional area.    #
#############################################################


class FrameData:
    load = 0    # Load in N
    disp = 0    # Displacement in mm
    stress = 0  # Stress in MPa

    def __init__(self, load, disp, stress):
        self.load = load
        self.disp = disp
        self.stress = stress


class SpecimenDimensions:
    w = 0   # Specimen Width in mm
    t = 0   # Specimen Thickness in mm
    gl = 0  # Specimen Gauge Length in mm
    ol = 0  # Specimen Overall Length in mm


def read_file(filepath):
    file = open(filepath, "r")

    specimen = SpecimenDimensions()
    frame_data = []
    frame_data.append(None)

    data_start = -1
    in_frame_data = False

    # Read lines of file into specimen variables and frame data
    for line in file:
        if "Width" in line:
            index = line.find("\t")
            specimen.w = float(line[index + 1:])

        if "Thickness" in line:
            index = line.find("\t")
            specimen.t = float(line[index + 1:])

        if "Gauge Length" in line:
            index = line.find("\t")
            specimen.gl = float(line[index + 1:])

        if "Overall Length" in line:
            index = line.find("\t")
            specimen.ol = float(line[index + 1:])

        if "Frame #" in line:
            data_start = 2

        if data_start != -1:
            data_start -= 1

        if data_start == 0:
            in_frame_data = True

        if in_frame_data:
            if line != '\t\t\n':
                frame_num_str, load_str, disp_str = line.split("\t")

                load = float(load_str)
                disp = float(disp_str)
                stress = load / (specimen.w * specimen.t)

                frame_data.append(FrameData(load, disp, stress))

    file.close()

    return specimen, frame_data
