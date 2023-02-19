## Requirements

Requirements are listed in requirements.txt file in root directory
Install all necessary libraries by running 

	$ pip install -r requirements.txt

## Launching

Launch the application by running 

	$ python main.py

From root directory

## Configuration

To change configuration of the application, modify "config.json" file located in "config" folder.

### Configuration parameters

#### feedback_mode 
	1 : Simple feedback. Oversteer and understeer values are given for 3 types of corners (slow speed corners, medium speed corners, high speed corners).
	2 : Comprehensive feedback. Oversteer and understeer values are given for each corner individually.

#### track
Represents the racing track. 2 tracks are currently supported:

	"imola"
	"spa"

#### steer_ratio
Represents car's steer ratio (ratio of turning angle of steering wheel to car's front wheels' angle of rotation)

	Default value : 14
