from math import pi, acos, sqrt
from decimal import Decimal, getcontext

getcontext().prec = 30

class Vector(object):

	CANNOT_NORMALIZE_ZERO_VECTOR_MSG = 'Cannot normalize the zero vector'
	NO_UNIQUE_PARALLEL_COMPONENT_MSG = 'no unique parallel component'
	ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG = 'only defined in two three dimensions'

	def __init__(self, coordinates):
		try:
			if not coordinates:
				raise ValueError
			self.coordinates = tuple([Decimal(x) for x in coordinates])
			self.dimension = len(coordinates)

		except ValueError:
			raise ValueError('The coordinates must be nonempty')
		except TypeError:
			raise TypeError('The coordinates must be iterable')

	def __str__(self):
		return 'Vector: {}'.format(self.coordinates)

	def __eq__(self, v):
		return self.coordinates == v.coordinates

	def __getitem__(self, item):
		return self.coordinates[item]

	def plus(self, v):
		new_coordinates = [x+y for x,y in zip(self.coordinates, v.coordinates)]
		return Vector(new_coordinates)

	def minus(self, v):
		new_coordinates = [x-y for x,y in zip(self.coordinates, v.coordinates)]
		return Vector(new_coordinates)

	def times_scalar(self, scalar):
		new_coordinates = [x*scalar for x in self.coordinates]
		return Vector(new_coordinates)

	def dot(self, v):
		new_coordinates = [x*y for x,y in zip(self.coordinates, v.coordinates)]
		return sum(new_coordinates)

	def magnitude(self):
		mag = [x**2 for x in self.coordinates]
		return Decimal(sqrt(sum(mag)))

	def normalized(self):
		try:
			return self.times_scalar(Decimal('1.0')/self.magnitude())
		except ZeroDivisionError:
			raise Exception('Cannot normalize the zero vector')

	def angle_with(self, v, in_degrees=False):
		v1 = self.normalized()
		v2 = v.normalized()
		v_dot = v1.dot(v2)

		angle_rad = acos(v_dot)
		if not in_degrees:
			return angle_rad
		else:
			degree_per_radian = 180 /pi
			return angle_rad * degree_per_radian

	def is_orthogonal_to(self, v, tolerance=1e-10):
		return abs(self.dot(v)) < tolerance

	def is_zero(self, tolerance=1e-10):
		return self.magnitude() < tolerance

	def is_parallel_to(self, v):
		return (self.is_zero() or
				v.is_zero() or
				self.angle_with(v) == Decimal('0') or
				self.angle_with(v) == pi)

	def component_parallel_to(self, basis):
		try:
			u = basis.normalized()
			weight = self.dot(u)
			return u.times_scalar(weight)
		except Exception as e:
			if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
				raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT_MSG)
			else:
				raise e

	def component_orthogonal_to(self, basis):
		try:
			projection = self.component_parallel_to(basis)
			return self.minus(projection)
		except Exception as e:
			if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
				raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT_MSG)
			else:
				raise e

	def cross(self, v):
		try:
			x_1, y_1, z_1 = self.coordinates
			x_2, y_2, z_2 = v.coordinates
			new_coordinates = [ y_1*z_2 - y_2*z_1,
								-(x_1*z_2 - x_2*z_1),
								x_1*y_2 - x_2*y_1]
			return Vector(new_coordinates)
		except Exception as e:
			msg = str(e)
			if msg == 'need more than 2 values to unpack':
				self_embedded_in_R3 = Vector(self.coordinates + ('0', ))
				v_embedded_in_R3 = Vector(v.coordinates + ('0',))
				return self_embedded_in_R3.cross(v_embedded_in_R3)
			elif (msg == 'too many values to unpack' or
			      msg == 'need more than 1 value to unpack'):
				raise Exception(self.ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG)
			else:
				raise e

	def area_of_parallelogram_with(self, v):
		cross_product = self.cross(v)
		return cross_product.magnitude()

	def area_of_triangle_with(self, v):
		return self.area_of_parallelogram_with(v) / Decimal('2.0')


if __name__ == '__main__':
	# a = Vector(['1', '2', '3'])
	# b = Vector(['1', '2', '3'])
	# c = Vector(['3.00', '4.00', '5.00'])
	# d = Vector(['1.00', '2.00'])


	# print a
	# print b
	# print c
	# print d

	# print a==b
	# print a==c
    
	# print a.plus(c)
	# print a.minus(c)
	# print a.minus(b)

	# print a.times_scalar(3)

	# print a.dot(b)

	# print d.magnitude()
	# print d.normalized()
	# print a.angle_with(b)

	# v1 = Vector(['2'])
	# v2 = Vector(['2'])
	# print v1.normalized()
	# print v2.normalized()
	# print v1.dot(v2)
	# print acos(v1.dot(v2))

	# w1 = Vector(['3', '0', '0'])
	# w2 = Vector(['0', '4', '0'])
	# w3 = Vector(['-3', '0', '0'])

	# w2 = Vector(['-3', '0', '0'])

	# print w1
	# print w2
	# print w1.normalized()
	# print w2.normalized()
	# print w1.dot(w2)
	# print w1.normalized().dot(w2.normalized())
	# print w1.angle_with(w2)
	# print w1.angle_with(w2, True)
	# print w1.is_orthogonal_to(w2)
	# print w1.is_parallel_to(w2)
	# print w1.is_parallel_to(w3)


	v = Vector(['3', '4'])
	b = Vector(['3', '0'])

	print (v.component_parallel_to(b))
	print (v.component_orthogonal_to(b))
	print (v.cross(b))
	print (v.area_of_parallelogram_with(b))
	print (v.area_of_triangle_with(b))


v = Vector(['3.183', '-7.627'])
b = Vector(['-2.668', '5.319'])
print v.angle_with(b)

v = Vector(['7.35', '0.221', '5.188'])
w = Vector(['2.751', '8.259', '3.985'])
print v.angle_with(w, in_degrees=True)