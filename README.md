# Gaze-based Pedestrian Warning System

This system alerts the pedestrian in real-time when they fail to notice an approaching car in their way. It uses a head-mounted eye-tracker Tobii Pro Glasses 2 with a monocular scene camera and infrared (IR) eye cameras to stream video of the pedestrian’s point of view (POV) and their gaze point in 2D pixel coordinates relative to the video feed, respectively. Auditory warning is given to the pedestrian when they had not continuously looked at the approaching vehicle for at least 300 milliseconds. It is assumed that the repo is stored in the folder 'ped-gaze'. Terminal commands assume Ubuntu 18.04. 

# Citation

<add paper details>

# Setup

The code is written in Python. The system is tested with Python 3.6.9. To setup the environment run these two commands in a parent folder of the downloaded repository (replace `/` with `\` and possibly add `--user` if on Windows):
- `pip install -e ped-gaze` will setup the project as a package accessible in the environment. 
- `pip install -r ped-gaze/requirement.txt` will install the the required packages.
 
# Analysis

Code for analysis is written in Python. No additional packages are needed. To run, the analysis code, create a copy of the output from the terminal and run the file <>. This convers the terminal output into an .xls file which is used in data processing. Run the file <> to generate the accuracy and bounding box area plots as shown below. Visualizations are saved in the parent folder. 



## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
