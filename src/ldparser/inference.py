import numpy as np

class FuzzyClass:
	@classmethod
	def from_values(cls, membership, *args):
		return cls(*args)

	@classmethod
	def from_range(cls, membership, min_val=0, max_val=1, step=1):
		domain_list = np.arange(min_val, max_val+step, step)
		return cls(membership, domain_list)

	def get_membership_val(self, domain_val):
		return fuzzy_map[domain_val]


class FuzzyActivation:
	@staticmethod
	def activate_linear(input_val, domain, a_p, b_p):
		first = domain[0]
		last = domain[-1]
		val_range = (last-first)
		a_val = first + lower_b * val_range
		b_val = first + upper_b * val_range

		if input_val <= a_val: return 0;
		elif input_val > a_val and input_val < b_val:
			return (input_val - a_val) / (b_val - a_val)
		else: return 1


	@staticmethod
	def activate_inv_linear(input_val, domain, a_p, b_p):
		first = domain[0]
		last = domain[-1]
		val_range = (last-first)
		a_val = first + lower_b * val_range
		b_val = first + upper_b * val_range

		if input_val <= a_val: return 1;
		elif input_val > a_val and input_val < b_val:
			return (b_val - input_val) / (b_val - a_val)
		else: return 0

	@staticmethod
	def activate_triangular(input_val, domain, a_p, b_p, c_p):
		first = domain[0]
		last = domain[-1]
		val_range = (last-first)
		a_val = first + a_p * val_range
		b_val = first + b_p * val_range
		c_val = first + c_p * val_range

		if input_val <= a_val: return 0;
		elif input_val > a_val and input_val <= b_val:
			return (b_val - input_val) / (b_val - a_val)
		elif input_val > b_val and input_val < c_val:
			return (input_val - b_val) / (c_val - b_val)
		else: return 0

	@staticmethod
	def activate_linear(input_val, domain, a_p, b_p, c_p, d_p):
		first = domain[0]
		last = domain[-1]
		val_range = (last-first)
		a_val = first + a_p * val_range
		b_val = first + b_p * val_range
		c_val = first + c_p * val_range
		d_val = first + c_p * val_range

		if input_val <= a_val: return 0;
		elif input_val > a_val and input_val < b_val:
			return (b_val - input_val) / (b_val - a_val)
		elif input_val >= b_val and input_val <= c_val:
			return 1
		elif input_val > c_val and input_val < d_val:
			return (input_val - c_val) / (d_val - c_val)
		else: return 0



	@staticmethod
	def activate_tanh(input_val, domain, lower_b, upper_b, tanh_max=2.4):
		first = domain[0]
		last = domain[-1]
		val_range = (last-first)
		lower_val = first + lower_b * val_range
		upper_val = first + upper_b * val_range
		val_range = upper_val - lower_val
		
		if input_val <= lower_val: return 0;
		elif input_val > lower_val and input_val < upper_val:
			return (np.tanh((domain - first) / val_range * 2 * tanh_max - tanh_max) + 1) / 2
		else: return 1

		
	@staticmethod
	def activate_cos(input_val, domain, lower_b, upper_b):
		first = domain[0]
		last = domain[-1]
		val_range = (last-first)
		lower_val = first + lower_b * val_range
		upper_val = first + upper_b * val_range
		val_range = upper_val - lower_val
		
		if input_val <= lower_val: return 0;
		elif input_val > lower_val and input_val < upper_val:
			return (np.cos((input_val - lower_val) / val_range * np.pi + np.pi) + 1) / 2
		else: return 1

	@staticmethod
	def activate_sin(input_val, domain, lower_b, upper_b):
		first = domain[0]
		last = domain[-1]
		val_range = (last-first)
		lower_val = first + lower_b * val_range
		upper_val = first + upper_b * val_range
		val_range = upper_val - lower_val
		
		if input_val <= lower_val: return 0;
		elif input_val > lower_val and input_val < upper_val:
			return np.sin((input_val - lower_val) / val_range * np.pi)
		else: return 0

		


class FuzzyVariable:
	def __init__(self, input_val, domain):
		self.input_val = input_val
		self.domain = domain
		self.sets = {

		}

	@classmethod
	def from_values(cls, input_val, *args):
		return cls(*args)

	@classmethod
	def from_range(cls, input_val, min_val=0, max_val=1, step=1):
		domain_list = np.arange(min_val, max_val+step, step)
		return cls(input_val, domain_list)

	@classmethod
	def from_fuzzy_class(cls, input_val, fuzzy_cls):
		return cls(input_val, fuzzy_cls.domain)

	#TODO test
	def add_set_sin(self, name, lower_percentile, upper_percentile):
		if name in self.sets: raise ValueError("Such fuzzy set already exists")
		self.sets[name] = FuzzyActivation.activate_sin(self.input_val, self.domain, lower_percentile, upper_percentile)

	#TODO test
	def add_set_cos(self, name, lower_percentile, upper_percentile):
		if name in self.sets: raise ValueError("Such fuzzy set already exists")
		self.sets[name] = FuzzyActivation.activate_cos(self.input_val, self.domain, lower_percentile, upper_percentile)

	#TODO
	def add_set_tanh(self, name, lower_percentile, upper_percentile):
		if name in self.sets: raise ValueError("Such fuzzy set already exists")
		self.sets[name] = FuzzyActivation.activate_sin(self.input_val, self.domain, lower_percentile, upper_percentile)

	#TODO test
	def add_set_linear(self, name, a, b):
		if name in self.sets: raise ValueError("Such fuzzy set already exists")
		self.sets[name] = FuzzyActivation.activate_sin(self.input_val, self.domain, lower_percentile, upper_percentile)

	#TODO test
	def add_set_inv_linear(self, name, a, b):
		if name in self.sets: raise ValueError("Such fuzzy set already exists")
		self.sets[name] = FuzzyActivation.activate_sin(self.input_val, self.domain, lower_percentile, upper_percentile)

	#TODO test
	def add_set_triangular(self, name, a, b, c):
		if name in self.sets: raise ValueError("Such fuzzy set already exists")
		self.sets[name] = FuzzyActivation.activate_sin(self.input_val, self.domain, lower_percentile, upper_percentile)

	#TODO test
	def add_set_trapesoidal(self, name, a, b, c, d):
		if name in self.sets: raise ValueError("Such fuzzy set already exists")
		self.sets[name] = FuzzyActivation.activate_sin(self.input_val, self.domain, lower_percentile, upper_percentile)

class FuzzySet:
	def __init__(self, membership_val):
		self.membership_val = membership_val

	def AND(self, fuzzy_set):
		self.membership_val = np.minimum(self.membership_val, fuzzy_set.membership_val)
		return self

	def OR(self, fuzzy_set):
		self.membership_val = np.maximum(self.membership_val, fuzzy_set.membership_val)
		return self

	def NOT(self):
		self.membership_val = 1 - self.membership_val
		return self

if __name__ == "__main__":
	fuzzy_var = FuzzyVariable.from_range(175, 100, 200, 5)
	fuzzy_var.add_set_sin("test", 0.25, 0.75)
	print(fuzzy_var.sets["test"])



















