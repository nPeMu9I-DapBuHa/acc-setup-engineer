import json
from enum import Enum
import numpy as np

class Setup:
	def __init__(self, data):
		self.car_name = data["carName"]
		self.data = data
		self.possible_actions = self.get_possible_actions()
		self.normalized_data = self.normalize_setup()

	@classmethod
	def from_file(cls, file_path):
		with open(file_path) as f:
  			return cls(json.load(f))

	def normalize_setup(self):
		setup = self.data
		new_setup = {}

		basic = setup["basicSetup"]
		wheels = basic["alignment"]

		camber = wheels["camber"]
		front = camber[0]
		rear = camber[2]
		new_setup["REAR_CAMBER"] = rear / 25
		new_setup["FRONT_CAMBER"] = front / 25

		toe = wheels["toe"]
		front = toe[0]
		rear = toe[2]

		new_setup["REAR_TOE"] = rear / 80
		new_setup["FRONT_TOE"] = front / 80

		advanced = setup["advancedSetup"]
		grip = advanced["mechanicalBalance"]
		farb = grip["aRBFront"]

		new_setup["FRONT_ANTIROLL_BAR"] = farb / 49

		rarb = grip["aRBRear"]

		new_setup["REAR_ANTIROLL_BAR"] = rarb / 49

		brake_bias = grip["brakeBias"]

		new_setup["REAR_ANTIROLL_BAR"] = brake_bias / 105

		bumpstop_range = grip["bumpStopWindow"]
		front = bumpstop_range[0]
		rear = bumpstop_range[2]


		new_setup["REAR_BUMPSTOP_RANGE"] = rear / 60
		new_setup["FRONT_BUMPSTOP_RANGE"] = front / 19

		bumpstop_rate = grip["bumpStopRateUp"]
		front = bumpstop_rate[0]
		rear = bumpstop_rate[2]

		new_setup["REAR_BUMPSTOP_RATE"] = rear / 22
		new_setup["FRONT_BUMPSTOP_RATE"] = front / 22

		spring = grip["wheelRate"]
		front = spring[0]
		rear = spring[2]

		new_setup["REAR_SPRING_RATE"] = rear / 10
		new_setup["FRONT_SPRING_RATE"] = front / 10

		drivetrain = advanced["drivetrain"]
		preload = drivetrain["preload"]

		new_setup["DIFFERENTIAL_PRELOAD"] = front / 28

		aeroBalance = advanced["aeroBalance"]
		rideHeight = aeroBalance["rideHeight"]
		front = rideHeight[0]
		rear = rideHeight[2]

		new_setup["REAR_RIDE_HEIGHT"] = rear / 25
		new_setup["FRONT_RIDE_HEIGHT"] = front / 25

		rearWing = aeroBalance["rearWing"]
		new_setup["REAR_WING"] = rearWing / 14

		return new_setup


	def get_possible_actions(self):
		setup = self.data
		possible_actions = []

		basic = setup["basicSetup"]
		wheels = basic["alignment"]

		camber = wheels["camber"]
		front = camber[0]
		rear = camber[2]
		if front < 25:
			possible_actions.append(Actions.FRONT_CAMBER_UP)
		if front > 0:
			possible_actions.append(Actions.FRONT_CAMBER_DOWN)
		if rear < 25:
			possible_actions.append(Actions.REAR_CAMBER_UP)
		if rear > 0:
			possible_actions.append(Actions.REAR_CAMBER_DOWN)

		toe = wheels["toe"]
		front = toe[0]
		rear = toe[2]
		if front < 80:
			possible_actions.append(Actions.FRONT_TOE_UP)
		if front > 0:
			possible_actions.append(Actions.FRONT_TOE_DOWN)
		if rear < 80:
			possible_actions.append(Actions.REAR_TOE_UP)
		if rear > 0:
			possible_actions.append(Actions.REAR_TOE_DOWN)

		advanced = setup["advancedSetup"]
		grip = advanced["mechanicalBalance"]
		farb = grip["aRBFront"]
		if farb < 49:
			possible_actions.append(Actions.FRONT_ANTIROLL_BAR_UP)
		if farb > 0:
			possible_actions.append(Actions.FRONT_ANTIROLL_BAR_DOWN)

		rarb = grip["aRBRear"]
		if rarb < 49:
			possible_actions.append(Actions.REAR_ANTIROLL_BAR_UP)
		if rarb > 0:
			possible_actions.append(Actions.REAR_ANTIROLL_BAR_DOWN)

		brake_bias = grip["brakeBias"]
		if brake_bias < 105:
			possible_actions.append(Actions.BRAKE_BIAS_UP)
		if brake_bias > 0:
			possible_actions.append(Actions.BRAKE_BIAS_DOWN)

		bumpstop_range = grip["bumpStopWindow"]
		front = bumpstop_range[0]
		rear = bumpstop_range[2]
		if front < 19:
			possible_actions.append(Actions.FRONT_BUMPSTOP_RANGE_UP)
		if front > 0:
			possible_actions.append(Actions.FRONT_BUMPSTOP_RANGE_DOWN)
		if rear < 60:
			possible_actions.append(Actions.REAR_BUMPSTOP_RANGE_UP)
		if rear > 0:
			possible_actions.append(Actions.REAR_BUMPSTOP_RANGE_DOWN)

		bumpstop_rate = grip["bumpStopRateUp"]
		front = bumpstop_rate[0]
		rear = bumpstop_rate[2]
		if front < 22:
			possible_actions.append(Actions.FRONT_BUMPSTOP_RATE_UP)
		if front > 0:
			possible_actions.append(Actions.FRONT_BUMPSTOP_RATE_DOWN)
		if rear < 22:
			possible_actions.append(Actions.REAR_BUMPSTOP_RATE_UP)
		if rear > 0:
			possible_actions.append(Actions.REAR_BUMPSTOP_RATE_DOWN)

		spring = grip["wheelRate"]
		front = spring[0]
		rear = spring[2]
		if front < 10:
			possible_actions.append(Actions.FRONT_SPRING_RATE_UP)
		if front > 0:
			possible_actions.append(Actions.FRONT_SPRING_RATE_DOWN)
		if rear < 10:
			possible_actions.append(Actions.REAR_SPRING_RATE_UP)
		if rear > 0:
			possible_actions.append(Actions.REAR_SPRING_RATE_DOWN)

		drivetrain = advanced["drivetrain"]
		preload = drivetrain["preload"]
		if preload < 28:
			possible_actions.append(Actions.DIFFERENTIAL_PRELOAD_UP)
		if preload > 0:
			possible_actions.append(Actions.DIFFERENTIAL_PRELOAD_DOWN)

		aeroBalance = advanced["aeroBalance"]
		rideHeight = aeroBalance["rideHeight"]
		front = rideHeight[0]
		rear = rideHeight[2]
		if front < 25:
			possible_actions.append(Actions.FRONT_RIDE_HEIGHT_UP)
		if front > 0:
			possible_actions.append(Actions.FRONT_RIDE_HEIGHT_DOWN)
		if rear < 25:
			possible_actions.append(Actions.REAR_RIDE_HEIGHT_UP)
		if rear > 0:
			possible_actions.append(Actions.REAR_RIDE_HEIGHT_DOWN)

		rearWing = aeroBalance["rearWing"]
		if rearWing < 14:
			possible_actions.append(Actions.REAR_WING_UP)
		if rearWing > 0:
			possible_actions.append(Actions.REAR_WING_DOWN)

		possible_actions.sort()
		return np.array(possible_actions)


class Actions(int, Enum):
	FRONT_TOE_DOWN = 0
	FRONT_TOE_UP = 1
	FRONT_CAMBER_DOWN = 2
	FRONT_CAMBER_UP = 3
	REAR_TOE_DOWN = 4
	REAR_TOE_UP = 5
	REAR_CAMBER_DOWN = 6
	REAR_CAMBER_UP = 7

	FRONT_ANTIROLL_BAR_DOWN = 8
	FRONT_ANTIROLL_BAR_UP = 9
	BRAKE_BIAS_DOWN = 10
	BRAKE_BIAS_UP = 11
	FRONT_SPRING_RATE_DOWN = 12
	FRONT_SPRING_RATE_UP = 13
	FRONT_BUMPSTOP_RATE_DOWN = 14
	FRONT_BUMPSTOP_RATE_UP = 15
	FRONT_BUMPSTOP_RANGE_DOWN = 16
	FRONT_BUMPSTOP_RANGE_UP = 17
	REAR_ANTIROLL_BAR_DOWN = 18
	REAR_ANTIROLL_BAR_UP = 19
	DIFFERENTIAL_PRELOAD_DOWN = 20
	DIFFERENTIAL_PRELOAD_UP = 21
	REAR_SPRING_RATE_DOWN = 22
	REAR_SPRING_RATE_UP = 23
	REAR_BUMPSTOP_RATE_DOWN = 24
	REAR_BUMPSTOP_RATE_UP = 25
	REAR_BUMPSTOP_RANGE_DOWN = 26
	REAR_BUMPSTOP_RANGE_UP = 27
	REAR_WING_DOWN = 28
	REAR_WING_UP = 29
	REAR_RIDE_HEIGHT_DOWN = 30
	REAR_RIDE_HEIGHT_UP = 31
	FRONT_RIDE_HEIGHT_DOWN = 32
	FRONT_RIDE_HEIGHT_UP = 33

	