#-*-coding:utf-8-*-

from math import acos,pi,sqrt
from decimal import Decimal, getcontext

getcontext().prec = 10

class Vector(object):

    CANNOT_NORMALIZE_ZERO_VECTOR_MSG = 'Cannot compute an angle with the zero vector'
    NO_UNIQUE_PARALLEL_COMPONENT_MSG = 'NO_UNIQUE_PARALLEL_COMPONENT_MSG'
    NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG = 'NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG'

    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple([Decimal(x) for x in coordinates])
            self.dimension = Decimal(len(coordinates))

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')

    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)


    #为什么增加了这个函数就可以避免不能迭代的问题的呢？
    def __getitem__(self, item):
        return self.coordinates[item]
    #


    def dot(self, v):
        return sum(x*y for x,y in zip(self.coordinates, v.coordinates))

    def magnitude(self):
        coordinates_squard =[x**Decimal('2.0') for x in self.coordinates]
        return Decimal.sqrt(sum(coordinates_squard))

    def times_scalar(self, c):
        new_coordinates = [Decimal(c)*x for x in self.coordinates]
        return Vector(new_coordinates)

    def normalized(self):
        try:
            magnitude = self.magnitude()
            return self.times_scalar(Decimal(1)/Decimal(magnitude))
        except ZeroDivisionError:
            raise Exception('Cannot normalize the zero vector')

    def angle_with(self, v, in_degrees=False):
        try:
            u1 = self.normalized()
            u2 = v.normalized()
            angle_in_radians = acos(u1.dot(u2))

            if in_degrees:
                degrees_per_radian = Decimal(180)/Decimal(pi)
                return Decimal(angle_in_radians) * degrees_per_radian
            else:
                return angle_in_radians
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception('Cannot compute an angle with the zero vector')
            else:
                raise e

    def is_orthogonal_to(self, v, tolerance=1e-10):
        return abs(self.dot(v)) < tolerance

    def is_parallel_to(self, v):
        return (self.is_zero() or v.is_zero() or
                self.angle_with(v) == Decimal('0') or
                self.angle_with(v) == pi)

    def is_zero(self, tolerance=1e-10):
        return self.magnitude() < tolerance


#vertor_component_orthogonal.py



    def minus(self, v):
        new_coordinates = [x-y for x,y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)

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
            if str(e) == self.NO_UNIQUE_PARALLEL_COMPONENT_MSG:
                raise Exception(self.NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG)
            else:
                raise e


print '#1'
v = Vector([3.039, 1.879])
w = Vector([0.825, 2.036])
print v.component_parallel_to(w)

print '\n#2'
v = Vector([-9.88, -3.264, -8.159])
w = Vector([-2.155, -9.353, -9.473])
print v.component_orthogonal_to(w)

print '\n#3'
v = Vector([3.009, -6.172, 3.692, -2.51])
w = Vector([6.404, -9.144, 2.759, 8.718])
vpar = v.component_parallel_to(w)
vort = v.component_orthogonal_to(w)
print "parallel component:", vpar
print "orthogonal component:", vort



