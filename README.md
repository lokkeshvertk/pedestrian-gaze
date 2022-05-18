# Gaze-based Pedestrian Warning System

This system alerts the pedestrian in real-time when they fail to notice an approaching car in their way. It uses a head-mounted eye-tracker Tobii Pro Glasses 2 with a monocular scene camera and infrared (IR) eye cameras to stream video of the pedestrian’s point of view (POV) and their gaze point in 2D pixel coordinates relative to the video feed, respectively. Auditory warning is given to the pedestrian when they had not continuously looked at the approaching vehicle for at least 300 milliseconds. It is assumed that the repo is stored in the folder 'pedestrian-gaze'. The system is initiated by `cd ~/pedestrian-gaze && python3 gpws.py`. Terminal commands assume Ubuntu 18.04. 

# Citation

"add paper details"

# Setup

The code is written in Python. The system is tested with Python 3.6.9. To setup the environment run these two commands in a parent folder of the downloaded repository (replace `/` with `\` and possibly add `--user` if on Windows):
- `pip install -e ped-gaze` will setup the project as a package accessible in the environment. 
- `pip install -r ped-gaze/requirements.txt` will install the the required packages.
 
# Analysis

Code for analysis is written in Python. No additional packages are needed. To run, the analysis code, create a copy of the output from the terminal and run the file  `extract_terminal_data.py`. This convers the terminal output into an .xls file which is used in data processing and are saved in the parent folder. 

# Used Assets
| Name | Developer | License
| --- | --- | ---
|[Yolov4](https://github.com/AlexeyAB/darknet) | Bochkovskiy et al. (2020) | -
|[Monodepth2](https://github.com/nianticlabs/monodepth2) | Godard et al.(2019) | Niantic, Inc.2019
|[TobiiGlassesPyController](https://github.com/ddetommaso/TobiiGlassesPyController) | De Tommaso et al. (2019) | - 

