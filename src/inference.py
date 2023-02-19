import simpful as sf
import numpy as np

class InferenceSystem:
	def __init__(self):
		self.actions  = [
			"FRONT_TOE",
			"FRONT_CAMBER",
			"REAR_TOE",
			"REAR_CAMBER",
			"FRONT_ANTIROLL_BAR",
			"BRAKE_BIAS",
			"FRONT_SPRING_RATE",
			"FRONT_BUMPSTOP_RATE",
			"FRONT_BUMPSTOP_RANGE",
			"REAR_ANTIROLL_BAR",
			"DIFFERENTIAL_PRELOAD",
			"REAR_SPRING_RATE",
			"REAR_BUMPSTOP_RATE",
			"REAR_BUMPSTOP_RANGE",
			"REAR_WING",
			"REAR_RIDE_HEIGHT",
			"FRONT_RIDE_HEIGHT",
		]
		self.input_var = ["CORNER_ENTRY", "CORNER_MID", "CORNER_EXIT"]
		self.condition = ["understeer", "oversteer"]
		self.action_results = [
			[4, 4, 4, -4, -4, -4],#"FRONT_TOE",
			[-3, -5, -2, 3, 5, 2],#"FRONT_CAMBER",
			[-5, -5, -5, 5, 5, 5],#"REAR_TOE",
			[-5, -5, 0, 5, 5, 0],#"REAR_CAMBER",
			[-5, -3, 0, 5, 3, 0],#"FRONT_ANTIROLL_BAR",
			[-5, 0, 0, 5, 0, 0],#"BRAKE_BIAS",
			[-5, -2, 5, 5, 2, -5],#"FRONT_SPRING_RATE",
			[4, 0, 0, -4, 0, 0],#"FRONT_BUMPSTOP_RATE",
			[-5, 0, 0, 5, 0, 0],#"FRONT_BUMPSTOP_RANGE",
			[1, 3, 5, -1, -3, -5],#"REAR_ANTIROLL_BAR",
			[1, 5, 7, -1, -5, -7],#"DIFFERENTIAL_PRELOAD",
			[3, 5, 7, -3, -5, -7],#"REAR_SPRING_RATE",
			[0, 0, 5, 0, 0, -5],#"REAR_BUMPSTOP_RATE",
			[-5, 0, 0, 5, 0, 0],#"REAR_BUMPSTOP_RANGE",
			[7, -3, -7, 7, 3, -7],#"REAR_WING",
			[5, 5, 5, -5, -5, -5],#"REAR_RIDE_HEIGHT",
			[-7, -5, -2, 7, 5, 2],#"FRONT_RIDE_HEIGHT",
		]
		self.FS = sf.FuzzySystem()
		S_1 = sf.FuzzySet( points=[[-10, 0], [0, 0], [1, 0.15], [2, 0.29], [3, 0.43], [4, 0.56], [5, 0.67], [6, 0.77], [7, 0.85], [8, 0.91], [9, 0.96], [10, 1]],          term="oversteer")
		S_2 = sf.FuzzySet( points=[[-10, 1], [-9, 0.96], [-8, 0.91], [-7, 0.85], [-6, 0.77], [-5, 0.67], [-4, 0.56], [-3, 0.43], [-2, 0.29], [-1, 0.15], [0,  0], [10, 0]],          term="understeer")
		normal = sf.FuzzySet( points=[[-10, 0], [-0.5, 0.0], [0,  1], [0.5, 0], [10, 0]],          term="normal")
		S_3 = sf.FuzzySet( points=([0, 0], [0.1, 0], [0.9, 1], [1, 1]), term="high")
		front_stiff = sf.FuzzySet( points=([0, 0], [0.1, 0], [0.3, 0.1], [0.6, 0.75], [0.8, 0.95], [0.9, 1], [1, 1]), term="front_stiff")

		arb_soft = sf.FuzzySet( points=([0, 1], [0.2, 1], [0.9, 0], [1, 0]), term="soft")
		arb_stiff = sf.FuzzySet( points=([0, 1], [0.4, 0], [0.9, 1], [1, 0]), term="stiff")

		self.FS.add_linguistic_variable("SLOW_CORNER_ENTRY",   sf.LinguisticVariable( [S_1, S_2, normal] ))
		self.FS.add_linguistic_variable("SLOW_CORNER_MID",   sf.LinguisticVariable( [S_1, S_2, normal] ))
		self.FS.add_linguistic_variable("SLOW_CORNER_EXIT",   sf.LinguisticVariable( [S_1, S_2, normal] ))
		self.FS.add_linguistic_variable("MED_CORNER_ENTRY", sf.LinguisticVariable( [S_1, S_2, normal] ))
		self.FS.add_linguistic_variable("MED_CORNER_MID", sf.LinguisticVariable( [S_1, S_2, normal] ))
		self.FS.add_linguistic_variable("MED_CORNER_EXIT", sf.LinguisticVariable( [S_1, S_2, normal] ))
		self.FS.add_linguistic_variable("FAST_CORNER_ENTRY",   sf.LinguisticVariable( [S_1, S_2, normal] ))
		self.FS.add_linguistic_variable("FAST_CORNER_MID",   sf.LinguisticVariable( [S_1, S_2, normal] ))
		self.FS.add_linguistic_variable("FAST_CORNER_EXIT",   sf.LinguisticVariable( [S_1, S_2, normal] ))

		self.FS.add_linguistic_variable("CORNER_ENTRY",   sf.LinguisticVariable( [S_1, S_2, normal] ))
		self.FS.add_linguistic_variable("CORNER_MID",   sf.LinguisticVariable( [S_1, S_2, normal] ))
		self.FS.add_linguistic_variable("CORNER_EXIT",   sf.LinguisticVariable( [S_1, S_2, normal] ))


		self.FS.add_linguistic_variable("REAR_BUMPSTOP_RANGE_IN",   sf.LinguisticVariable( [S_3] ))
		self.FS.add_linguistic_variable("FRONT_SPRING_RATE_IN",   sf.LinguisticVariable( [front_stiff] ))
		self.FS.add_linguistic_variable("FRONT_ANTIROLL_BAR_IN",   sf.LinguisticVariable( [arb_soft, arb_stiff] ))

		arb_soft = sf.FuzzySet( points=([0, 1], [0.06, 1], [0.35, 0], [1, 0]), term="soft")
		arb_stiff = sf.FuzzySet( points=([0, 0], [0.06, 0], [0.35, 1], [1, 1]), term="stiff")
		self.FS.add_linguistic_variable("REAR_ANTIROLL_BAR_IN",   sf.LinguisticVariable( [arb_soft, arb_stiff] ))

		low = sf.FuzzySet( points=([0, 1], [0.3, 0], [1, 0]), term="low")
		high = sf.FuzzySet( points=([0, 0], [0.1, 0], [0.4, 1], [1, 1]), term="high")
		self.FS.add_linguistic_variable("FRONT_RIDE_HEIGHT_IN",   sf.LinguisticVariable( [low, high] ))

		low = sf.FuzzySet( points=([0, 1], [0.5, 0], [1, 0]), term="low")
		high = sf.FuzzySet( points=([0, 0], [0.2, 0], [0.6, 1], [1, 1]), term="high")
		self.FS.add_linguistic_variable("REAR_RIDE_HEIGHT_IN",   sf.LinguisticVariable( [low, high] ))

		low = sf.FuzzySet( points=([0, 1], [0.3, 1], [0.8, 0], [1, 0]), term="low")
		high = sf.FuzzySet( points=([0, 0], [0.3, 0], [0.8, 1], [1, 1]), term="high")
		self.FS.add_linguistic_variable("REAR_WING_IN",   sf.LinguisticVariable( [low, high] ))


		self.FS.set_crisp_output_value("NO_CHANGE", 0)
		self.FS.set_crisp_output_value("SLIGHT_CHANGE", 0.25)
		self.FS.set_crisp_output_value("CHANGE", 0.5)
		self.FS.set_crisp_output_value("MAJOR_CHANGE", 0.75)
		self.FS.set_crisp_output_value("DEFINITELY_CHANGE", 1)



		for i in range(-10, 11, 1):
			if i < 0:
				self.FS.set_crisp_output_value(f"M{-i}", i/10)
			else:
				self.FS.set_crisp_output_value(f"{i}", i/10)

		rules = [
			"IF (SLOW_CORNER_ENTRY IS understeer) AND (FAST_CORNER_ENTRY IS understeer) AND (FRONT_ANTIROLL_BAR_IN IS stiff) THEN (FRONT_ANTI_ROLL_BAR IS M5)",
			"IF (SLOW_CORNER_ENTRY IS understeer) AND (FAST_CORNER_ENTRY IS understeer) THEN (FRONT_TOE IS M5)",
			"IF (SLOW_CORNER_ENTRY IS understeer) AND (FAST_CORNER_ENTRY IS understeer) THEN (DIFFERENTIAL_PRELOAD IS M5)",
			"IF (SLOW_CORNER_ENTRY IS understeer) AND ((MED_CORNER_ENTRY IS understeer) OR (FAST_CORNER_ENTRY IS understeer)) AND (FRONT_SPRING_RATE_IN IS front_stiff) THEN (FRONT_SPRING_RATE IS M5)",
			"IF (MED_CORNER_ENTRY IS understeer) AND (FAST_CORNER_ENTRY IS understeer) THEN (REAR_WING IS M5)",
			"IF (SLOW_CORNER_MID IS understeer) AND (SLOW_CORNER_ENTRY IS understeer) AND ((FRONT_RIDE_HEIGHT_IN IS high) OR (REAR_RIDE_HEIGHT_IN IS low)) THEN (REAR_RIDE_HEIGHT IS 9)",
			"IF (MED_CORNER_MID IS understeer) AND (MED_CORNER_ENTRY IS understeer) AND ((FRONT_RIDE_HEIGHT_IN IS high) OR (REAR_RIDE_HEIGHT_IN IS low)) THEN (REAR_RIDE_HEIGHT IS 5)",
			"IF (FAST_CORNER_MID IS understeer) AND (FAST_CORNER_ENTRY IS understeer) AND ((FRONT_RIDE_HEIGHT_IN IS high) OR (REAR_RIDE_HEIGHT_IN IS low)) THEN (REAR_RIDE_HEIGHT IS 1)",
			"IF (SLOW_CORNER_MID IS oversteer) AND (SLOW_CORNER_ENTRY IS oversteer) AND ((FRONT_RIDE_HEIGHT_IN IS low) OR (REAR_RIDE_HEIGHT_IN IS high)) THEN (REAR_RIDE_HEIGHT IS M9)",
			"IF (MED_CORNER_MID IS oversteer) AND (MED_CORNER_ENTRY IS oversteer) AND ((FRONT_RIDE_HEIGHT_IN IS low) OR (REAR_RIDE_HEIGHT_IN IS high)) THEN (REAR_RIDE_HEIGHT IS M5)",
			"IF (FAST_CORNER_MID IS oversteer) AND (FAST_CORNER_ENTRY IS oversteer) AND ((FRONT_RIDE_HEIGHT_IN IS low) OR (REAR_RIDE_HEIGHT_IN IS high)) THEN (REAR_RIDE_HEIGHT IS M1)",

			"IF ((SLOW_CORNER_ENTRY IS understeer) OR (SLOW_CORNER_MID IS understeer) AND (SLOW_CORNER_EXIT IS understeer)) AND \
			((MED_CORNER_ENTRY IS understeer) OR (MED_CORNER_MID IS understeer) AND (MED_CORNER_EXIT IS understeer)) AND \
			((FAST_CORNER_ENTRY IS understeer) OR (FAST_CORNER_MID IS understeer) AND (FAST_CORNER_EXIT IS understeer)) THEN (REAR_TOE IS M5)",
			"IF ((SLOW_CORNER_ENTRY IS oversteer) OR (SLOW_CORNER_MID IS oversteer) AND (SLOW_CORNER_EXIT IS oversteer)) AND \
			((MED_CORNER_ENTRY IS oversteer) OR (MED_CORNER_MID IS oversteer) AND (MED_CORNER_EXIT IS oversteer)) AND \
			((FAST_CORNER_ENTRY IS oversteer) OR (FAST_CORNER_MID IS oversteer) AND (FAST_CORNER_EXIT IS oversteer)) THEN (REAR_TOE IS 5)",

			"IF (SLOW_CORNER_MID IS oversteer) OR (SLOW_CORNER_ENTRY IS oversteer) AND ((FRONT_RIDE_HEIGHT_IN IS low) OR (REAR_RIDE_HEIGHT_IN IS high)) THEN (FRONT_RIDE_HEIGHT IS 9)"

			"IF ((SLOW_CORNER_ENTRY IS oversteer) OR (MED_CORNER_ENTRY IS oversteer) OR (FAST_CORNER_ENTRY IS oversteer)) AND (FRONT_ANTIROLL_BAR_IN IS soft) THEN (FRONT_ANTI_ROLL_BAR IS 5)",
			"IF (SLOW_CORNER_ENTRY IS oversteer) AND (MED_CORNER_ENTRY IS oversteer) AND (FAST_CORNER_ENTRY IS oversteer) THEN (FRONT_TOE IS 5)",
			"IF (SLOW_CORNER_ENTRY IS oversteer) AND (MED_CORNER_ENTRY IS oversteer) AND (FAST_CORNER_ENTRY IS oversteer) THEN (DIFFERENTIAL_PRELOAD IS 5)",
			"IF (SLOW_CORNER_ENTRY IS oversteer) AND (MED_CORNER_ENTRY IS oversteer) AND (FAST_CORNER_ENTRY IS oversteer) THEN (FRONT_SPRING_RATE IS 5)",
			"IF (MED_CORNER_ENTRY IS oversteer) AND (FAST_CORNER_ENTRY IS oversteer) THEN (REAR_WING IS 5)",


			"IF ((SLOW_CORNER_ENTRY IS understeer) OR (MED_CORNER_ENTRY IS understeer) OR (FAST_CORNER_ENTRY IS understeer)) AND (REAR_ANTIROLL_BAR_IN IS soft) THEN (REAR_ANTI_ROLL_BAR IS 5)",
			"IF ((SLOW_CORNER_EXIT IS understeer) OR (MED_CORNER_EXIT IS understeer)) AND (FAST_CORNER_EXIT IS understeer) THEN (DIFFERENTIAL_PRELOAD IS 5)",
			"IF ((SLOW_CORNER_EXIT IS understeer) OR (MED_CORNER_EXIT IS understeer)) AND (FAST_CORNER_EXIT IS understeer) THEN (REAR_SPRING_RATE IS 7)",

			"IF ((SLOW_CORNER_EXIT IS oversteer) OR (MED_CORNER_EXIT IS oversteer)) AND (FAST_CORNER_EXIT IS oversteer) AND (REAR_ANTIROLL_BAR_IN IS stiff) THEN (REAR_ANTI_ROLL_BAR IS M7)", # check this rule
			"IF ((SLOW_CORNER_EXIT IS oversteer) OR (MED_CORNER_EXIT IS oversteer)) AND (FAST_CORNER_EXIT IS oversteer) THEN (DIFFERENTIAL_PRELOAD IS M5)",
			"IF ((SLOW_CORNER_EXIT IS oversteer) OR (MED_CORNER_EXIT IS oversteer)) AND (FAST_CORNER_EXIT IS oversteer) THEN (REAR_SPRING_RATE IS M7)",

			"IF ((SLOW_CORNER_ENTRY IS understeer) AND (MED_CORNER_ENTRY IS understeer)) THEN (FRONT_BUMPSTOP_RANGE IS 5)",

			"IF ((SLOW_CORNER_EXIT IS understeer) OR (MED_CORNER_EXIT IS understeer)) AND (REAR_BUMPSTOP_RANGE_IN IS high) THEN (REAR_BUMPSTOP_RANGE IS M5)",
			"IF ((SLOW_CORNER_MID IS oversteer) OR (MED_CORNER_MID IS oversteer) OR (FAST_CORNER_MID IS oversteer)) AND \
			 ((SLOW_CORNER_EXIT IS oversteer) OR (MED_CORNER_EXIT IS oversteer) OR (FAST_CORNER_EXIT IS oversteer)) THEN (REAR_WING IS 5)",
			"IF ((SLOW_CORNER_MID IS understeer) OR (MED_CORNER_MID IS understeer) OR (FAST_CORNER_MID IS understeer)) AND \
			 ((SLOW_CORNER_EXIT IS understeer) OR (MED_CORNER_EXIT IS understeer) OR (FAST_CORNER_EXIT IS understeer)) THEN (REAR_WING IS M5)",

			"IF ((SLOW_CORNER_MID IS understeer) OR (MED_CORNER_MID IS understeer) OR (FAST_CORNER_MID IS understeer)) OR \
			 ((SLOW_CORNER_EXIT IS understeer) OR (MED_CORNER_EXIT IS understeer) OR (FAST_CORNER_EXIT IS understeer)) AND (REAR_RIDE_HEIGHT_IN IS high) THEN (REAR_RIDE_HEIGHT IS 5)",

			"IF ((SLOW_CORNER_MID IS understeer) OR (MED_CORNER_MID IS understeer) OR (FAST_CORNER_MID IS understeer)) OR \
			 ((SLOW_CORNER_EXIT IS understeer) OR (MED_CORNER_EXIT IS understeer) OR (FAST_CORNER_EXIT IS understeer)) THEN (REAR_WING IS M5)"
		]
		for i in range(17):
			for j, input_var in enumerate(self.input_var):
				for k, condition in enumerate(self.condition):
					if self.action_results[i][3*k+j] < 0:
						reward = "M" + str(np.absolute(self.action_results[i][3*k+j]))
					else:
						reward = self.action_results[i][3*k+j]
					rules.append(f"IF ({input_var} IS {condition}) THEN ({self.actions[i]} IS {reward})")

		self.FS.add_rules(rules)
		
	def inference(self, steer_profile, setup):
		return_dict = {}
		self.FS.set_variable("SLOW_CORNER_ENTRY", steer_profile[0])
		self.FS.set_variable("SLOW_CORNER_MID", steer_profile[1])
		self.FS.set_variable("SLOW_CORNER_EXIT", steer_profile[2])
		self.FS.set_variable("MED_CORNER_ENTRY", steer_profile[3])
		self.FS.set_variable("MED_CORNER_MID", steer_profile[4])
		self.FS.set_variable("MED_CORNER_EXIT", steer_profile[5])			
		self.FS.set_variable("FAST_CORNER_ENTRY", steer_profile[6])
		self.FS.set_variable("FAST_CORNER_MID", steer_profile[7])
		self.FS.set_variable("FAST_CORNER_EXIT", steer_profile[8])
		self.FS.set_variable("CORNER_ENTRY", (steer_profile[0] + steer_profile[3] + steer_profile[6]) / 3)
		self.FS.set_variable("CORNER_MID", (steer_profile[1] + steer_profile[4] + steer_profile[7]) / 3)
		self.FS.set_variable("CORNER_EXIT", (steer_profile[2] + steer_profile[5] + steer_profile[8]) / 3)

		self.FS.set_variable("REAR_BUMPSTOP_RANGE_IN", setup["REAR_BUMPSTOP_RANGE"])

		self.FS.set_variable("FRONT_SPRING_RATE_IN", setup["FRONT_SPRING_RATE"])
		self.FS.set_variable("FRONT_ANTIROLL_BAR_IN", setup["FRONT_ANTIROLL_BAR"])
		self.FS.set_variable("REAR_ANTIROLL_BAR_IN", setup["REAR_ANTIROLL_BAR"])

		self.FS.set_variable("FRONT_RIDE_HEIGHT_IN", setup["FRONT_RIDE_HEIGHT"])
		self.FS.set_variable("REAR_RIDE_HEIGHT_IN", setup["REAR_RIDE_HEIGHT"])
		self.FS.set_variable("REAR_WING_IN", setup["REAR_WING"])

		inferred_actions = self.FS.Sugeno_inference()
		results = {}
		for key, value in inferred_actions.items():
			results[key+"_UP"] = value
			results[key+"_DOWN"] = -value if value != 0.0 else 0.0

		return results

class DamperInferenceSystem():
	def __init__(self):
		self.FS = sf.FuzzySystem()
	
		soft = sf.FuzzySet(points=([-1, 1], [0.05, 1], [0.11, 0], [1, 0]), term="soft")
		balanced = sf.FuzzySet(points=([-1, 0], [0.05, 0], [0.11, 1], [0.16, 1], [0.20, 0], [1, 0]), term="balanced")
		stiff = sf.FuzzySet(points=([-1, 0], [0.13, 0], [0.20, 1], [1, 1]), term="stiff")
		
		bump_favoured = sf.FuzzySet(points=([-1, 1], [-0.12, 1], [0, 0], [1, 0]), term="bump_favoured")
		unfavoured = sf.FuzzySet(points=([-1, 0], [-0.12, 0], [0, 1], [0.12, 0], [1, 0]), term="unfavoured")
		rebound_favoured = sf.FuzzySet(points=([-1, 0], [0, 0], [0.12, 1], [1, 1]), term="rebound_favoured")

		normal = sf.FuzzySet(points=([0, 0], [0.40, 0], [0.52, 1], [0.6, 1], [0.75, 0]), term="normal")
		low = sf.FuzzySet(points=([0, 1], [0.25, 1], [0.55, 0], [1, 0]), term="low")
		high = sf.FuzzySet(points=([0, 0], [0.25, 0], [0.60, 1], [1, 1]), term="high")



		self.FS.add_linguistic_variable("MIDDLE_STIFFNESS",   sf.LinguisticVariable( [soft, balanced, stiff] ))
		self.FS.add_linguistic_variable("SLOW_DIFFERENCE", sf.LinguisticVariable( [bump_favoured, unfavoured, rebound_favoured] ))
		self.FS.add_linguistic_variable("SLOW_RATIO", sf.LinguisticVariable( [low, normal, high] ))

		normal = sf.FuzzySet(points=([0, 0], [0.7, 0], [0.9, 1], [1.11, 1], [1.5, 0], [100, 0]), term="normal")
		high = sf.FuzzySet(points=([0, 0], [1, 0], [2, 1], [100, 1]), term="high")
		self.FS.add_linguistic_variable("R_B", sf.LinguisticVariable( [normal, high] ))


		self.FS.add_linguistic_variable("B_R", sf.LinguisticVariable( [normal, high] ))

		self.FS.set_crisp_output_value("LOW", -1)
		self.FS.set_crisp_output_value("NORMAL", 0)
		self.FS.set_crisp_output_value("HIGH", 1)

		rules = [
			"IF (MIDDLE_STIFFNESS IS balanced) THEN (SPRING_RATE IS NORMAL)",
			"IF (MIDDLE_STIFFNESS IS stiff) AND (SLOW_DIFFERENCE IS unfavoured) THEN (SPRING_RATE IS LOW)",
			"IF (MIDDLE_STIFFNESS IS soft) AND (SLOW_DIFFERENCE IS unfavoured) THEN (SPRING_RATE IS HIGH)",
			"IF (SLOW_DIFFERENCE IS bump_favoured) AND (R_B IS normal) THEN (SLOW_BUMP IS HIGH)",
			"IF (SLOW_DIFFERENCE IS rebound_favoured) AND (B_R IS normal) THEN (SLOW_REBOUND IS HIGH)",
			"IF ((R_B IS high) AND (SLOW_DIFFERENCE IS unfavoured) AND (MIDDLE_STIFFNESS IS balanced)) THEN (SLOW_BUMP IS LOW)",
			"IF ((B_R IS high) AND (SLOW_DIFFERENCE IS unfavoured) AND (MIDDLE_STIFFNESS IS balanced)) THEN (SLOW_REBOUND IS LOW)",
			
			"IF (SLOW_RATIO IS normal) AND (SLOW_DIFFERENCE IS unfavoured) AND (R_B IS normal) THEN (SLOW_BUMP IS NORMAL)",
			"IF (SLOW_RATIO IS normal) AND (SLOW_DIFFERENCE IS unfavoured) AND (B_R IS normal) THEN (SLOW_REBOUND IS NORMAL)",
			# check rules for fast
			
			"IF (SLOW_RATIO IS high) AND (SLOW_DIFFERENCE IS unfavoured) THEN (FAST_BUMP IS NORMAL)",
			"IF (SLOW_RATIO IS high) AND (SLOW_DIFFERENCE IS unfavoured) THEN (FAST_REBOUND IS NORMAL)"
			"IF (SLOW_RATIO IS low) AND (SLOW_DIFFERENCE IS unfavoured) THEN (FAST_BUMP IS NORMAL)",
			"IF (SLOW_RATIO IS low) AND (SLOW_DIFFERENCE IS unfavoured) THEN (FAST_REBOUND IS NORMAL)"
		]
		self.FS.add_rules(rules)

	def inference(self, middle_stiffness, slow_difference, slow_ratio, b_r, r_b):
		self.FS.set_variable("MIDDLE_STIFFNESS", middle_stiffness)
		self.FS.set_variable("SLOW_DIFFERENCE", slow_difference)
		self.FS.set_variable("SLOW_RATIO", slow_ratio)

		self.FS.set_variable("B_R", b_r)
		self.FS.set_variable("R_B", r_b)


		return self.FS.Sugeno_inference()

class DamperOptimiser():
	def __init__(self):
		self.distribution = (0.22, 0.28, 0.28, 0.22)
		self.slow_to_fast_ratio_min = 2.8
		self.slow_to_fast_ratio_max = 3.5
		self.low_movement_mean_min = 0.11
		self.low_movement_mean_max = 0.13
		self.lf = DamperInferenceSystem()
		self.rf = DamperInferenceSystem()
		self.lr = DamperInferenceSystem()
		self.rr = DamperInferenceSystem()

	def infer(self, dampers):
		
		"""
		return {
            "lf": [lf_bump_slow, lf_bump_fast, lf_rebound_slow, lf_rebound_fast],
            "rf": [rf_bump_slow, rf_bump_fast, rf_rebound_slow, rf_rebound_fast],
            "lf": [lr_bump_slow, lr_bump_fast, lr_rebound_slow, lr_rebound_fast],
            "lf": [rr_bump_slow, rr_bump_fast, rr_rebound_slow, rr_rebound_fast]
        }
		"""
		#lf
		lf = dampers["lf"]
		low_movement_mean = np.sum(np.concatenate((lf[0][:4], lf[2][:4])))
		border_mean = np.mean(np.concatenate((lf[0][37:], lf[1][:3], lf[2][37:], lf[3][:3])))
		ratio = low_movement_mean / border_mean

		step1 = np.mean(lf[0][5:13])
		print(step1)
		step2 = np.mean(lf[0][13:21])
		print(step2)
		step3 = np.mean(lf[0][21:29])
		print(step3)
		stepness_bump = abs(np.mean((step1-step2, step2-step3)))

		print("rebound")
		step1 = np.mean(lf[2][5:13])
		print(step1)
		step2 = np.mean(lf[2][13:21])
		print(step2)
		step3 = np.mean(lf[2][21:29])
		print(step3)
		stepness_rebound = abs(np.mean((step1-step2, step2-step3)))

		b_r = stepness_bump / stepness_rebound
		r_b = stepness_rebound / stepness_bump

		print("lf")
		print("r_b" ,r_b)


		lf = self.lf.inference(low_movement_mean, np.sum(np.subtract(lf[2], lf[0])), np.sum(np.concatenate((lf[0], lf[2]))), b_r, r_b)

		#rf
		rf = dampers["rf"]
		low_movement_mean = np.sum(np.concatenate((rf[0][:4], rf[2][:4])))
		border_mean = np.mean(np.concatenate((rf[0][37:], rf[1][:3], rf[2][37:], rf[3][:3])))
		ratio = low_movement_mean / border_mean

		step1 = np.mean(rf[0][5:13])
		step2 = np.mean(rf[0][13:21])
		step3 = np.mean(rf[0][21:29])
		stepness_bump = abs(np.mean((step1-step2, step2-step3)))

		step1 = np.mean(rf[2][5:13])
		step2 = np.mean(rf[2][13:21])
		step3 = np.mean(rf[2][21:29])
		stepness_rebound = abs(np.mean((step1-step2, step2-step3)))

		b_r = stepness_bump / stepness_rebound
		r_b = stepness_rebound / stepness_bump

		rf = self.rf.inference(low_movement_mean, np.sum(np.subtract(rf[2], rf[0])), np.sum(np.concatenate((rf[0], rf[2]))), b_r, r_b)

		#lr
		lr = dampers["lr"]
		low_movement_mean = np.sum(np.concatenate((lr[0][:4], lr[2][:4])))
		border_mean = np.mean(np.concatenate((lr[0][37:], lr[1][:3], lr[2][37:], lr[3][:3])))
		ratio = low_movement_mean / border_mean

		step1 = np.mean(lr[0][5:13])
		step2 = np.mean(lr[0][13:21])
		step3 = np.mean(lr[0][21:29])
		stepness_bump = abs(np.mean((step1-step2, step2-step3)))

		step1 = np.mean(lr[2][5:13])
		step2 = np.mean(lr[2][13:21])
		step3 = np.mean(lr[2][21:29])
		stepness_rebound = abs(np.mean((step1-step2, step2-step3)))

		b_r = stepness_bump / stepness_rebound
		r_b = stepness_rebound / stepness_bump

		lr = self.lr.inference(low_movement_mean, np.sum(np.subtract(lr[2], lr[0])), np.sum(np.concatenate((lr[0], lr[2]))), b_r, r_b)

		#rr 
		rr = dampers["rr"]
		low_movement_mean = np.sum(np.concatenate((rr[0][:4], rr[2][:4])))
		border_mean = np.mean(np.concatenate((rr[0][37:], rr[1][:3], rr[2][37:], rr[3][:3])))
		ratio = low_movement_mean / border_mean

		step1 = np.mean(rr[0][5:13])
		step2 = np.mean(rr[0][13:21])
		step3 = np.mean(rr[0][21:29])
		stepness_bump = abs(np.mean((step1-step2, step2-step3)))

		step1 = np.mean(rr[2][5:13])
		step2 = np.mean(rr[2][13:21])
		step3 = np.mean(rr[2][21:29])
		stepness_rebound = abs(np.mean((step1-step2, step2-step3)))

		b_r = stepness_bump / stepness_rebound
		r_b = stepness_rebound / stepness_bump
		rr = self.rr.inference(low_movement_mean, np.sum(np.subtract(rr[2], rr[0])), np.sum(np.concatenate((rr[0], rr[2]))), b_r, r_b)
		return {
			"lf": lf,
			"rf": rf,
			"lr": lr,
			"rr": rr
		}




class SingleTyreOptimiser():

	def __init__(self, optimal_pressure_mean=27.6, optimal_pressure_variance=0.2, optimal_temperature_mean=82.5, optimal_temperature_variance=15):
		self.p_mean = optimal_pressure_mean
		self.p_var = optimal_pressure_variance
		self.t_mean = optimal_temperature_mean
		self.t_var = optimal_temperature_variance
		self.counter = 0
		self.mean_p_change = None


	def calc_pressure_delta(self, pressure):
		mean_p_change = self.get_mean_p_change()
		return np.rint((self.p_mean - pressure) / mean_p_change)

	def get_mean_p_change(self):
		if self.mean_p_change == None:
			return 0.1
		return self.mean_p_change


	def store_action(self, last_pressure, current_pressure, change):
		if change < 1 and change > -1:
			return 
		self.counter += 1
		if self.mean_p_change == None:
			self.mean_p_change = (current_pressure - last_pressure) / change
			return

		self.mean_p_change += (self.mean_p_change - (current_pressure - last_pressure) / change) / self.counter

	

class TyreOptimiser():
	def __init__(self, optimal_pressure_mean=27.6, optimal_pressure_variance=0.1, optimal_temperature_mean=82.5, optimal_temperature_variance=7.5):
		self.lf = SingleTyreOptimiser(optimal_pressure_mean, optimal_pressure_variance, optimal_temperature_mean, optimal_temperature_variance)
		self.rf = SingleTyreOptimiser(optimal_pressure_mean, optimal_pressure_variance, optimal_temperature_mean, optimal_temperature_variance)
		self.lr = SingleTyreOptimiser(optimal_pressure_mean, optimal_pressure_variance, optimal_temperature_mean, optimal_temperature_variance)
		self.rr = SingleTyreOptimiser(optimal_pressure_mean, optimal_pressure_variance, optimal_temperature_mean, optimal_temperature_variance)
		self.last_pressures = None
		self.last_action = None

	def infer(self, pressures):
		# lf
		steps = self.lf.calc_pressure_delta(pressures["lf"][1])
		print("Steps: ", steps)
		current_max_percentage = None
		lf_max_action = None
		for i in range(-60, 61, 1):
			current_steps = steps + i
			cur_pressure = np.copy(pressures["lf"][0]) + (current_steps * self.lf.get_mean_p_change())
			percentage = (np.count_nonzero(np.logical_and(cur_pressure>=self.lf.p_mean-self.lf.p_var, cur_pressure<=self.lf.p_mean+self.lf.p_var))) / cur_pressure.size
			if current_max_percentage is None or percentage > current_max_percentage:
				current_max_percentage = percentage
				lf_max_action = current_steps

		lf_max_action = int(np.rint((steps + lf_max_action) / 2))

		# rf
		steps = self.rf.calc_pressure_delta(pressures["rf"][1])
		print("Steps: ", steps)
		current_max_percentage = None
		rf_max_action = None
		for i in range(-60, 61, 1):
			current_steps = steps + i
			cur_pressure = pressures["rf"][0] + (current_steps * self.rf.get_mean_p_change())
			percentage = (np.count_nonzero(np.logical_and(cur_pressure>=self.rf.p_mean-self.rf.p_var, cur_pressure<=self.rf.p_mean+self.rf.p_var))) / cur_pressure.size
			if current_max_percentage is None or percentage > current_max_percentage:
				current_max_percentage = percentage
				rf_max_action = current_steps

		rf_max_action = int(np.rint((steps + rf_max_action) / 2))

		# lr
		steps = self.lr.calc_pressure_delta(pressures["lr"][1])
		print("Steps: ", steps)
		current_max_percentage = None
		lr_max_action = None
		for i in range(-60, 61, 1):
			current_steps = steps + i
			cur_pressure = pressures["lr"][0] + (current_steps * self.lr.get_mean_p_change())
			percentage = (np.count_nonzero(np.logical_and(cur_pressure>=self.lr.p_mean-self.lr.p_var, cur_pressure<=self.lr.p_mean+self.lr.p_var))) / cur_pressure.size
			print(current_steps, percentage)
			if current_max_percentage is None or percentage > current_max_percentage:
				current_max_percentage = percentage
				lr_max_action = current_steps

		lr_max_action = int(np.rint((steps + lr_max_action) / 2))
		print(lr_max_action)

		# rr
		steps = self.rr.calc_pressure_delta(pressures["rr"][1])
		print("Steps: ", steps)
		current_max_percentage = None
		rr_max_action = None
		for i in range(-60, 61, 1):
			current_steps = steps + i
			cur_pressure = pressures["rr"][0] + (current_steps * self.rr.get_mean_p_change())
			percentage = (np.count_nonzero(np.logical_and(cur_pressure>=self.rr.p_mean-self.rr.p_var, cur_pressure<=self.rr.p_mean+self.rr.p_var))) / cur_pressure.size
			if current_max_percentage is None or percentage > current_max_percentage:
				current_max_percentage = percentage
				rr_max_action = current_steps

		rr_max_action = int(np.rint((steps + rr_max_action) / 2))

		return {
			"lf": [self.lf.get_mean_p_change(), lf_max_action],
			"rf": [self.rf.get_mean_p_change(), rf_max_action],
			"lr": [self.lr.get_mean_p_change(), lr_max_action],
			"rr": [self.rr.get_mean_p_change(), rr_max_action]
		}

	def store_action(self, pressures, action):
		self.last_pressures = pressures
		self.last_action = action


	def store_results(self, pressures):
		if self.last_pressures is not None:
			self.lf.store_action(self.last_pressures["lf"][1], pressures["lf"][1], self.last_action["lf"])
			self.rf.store_action(self.last_pressures["rf"][1], pressures["rf"][1], self.last_action["rf"])
			self.lr.store_action(self.last_pressures["lr"][1], pressures["lr"][1], self.last_action["lr"])
			self.rr.store_action(self.last_pressures["rr"][1], pressures["rr"][1], self.last_action["rr"])
		self.last_pressures = None
	