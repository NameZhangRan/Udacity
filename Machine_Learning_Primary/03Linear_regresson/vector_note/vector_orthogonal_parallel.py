#-*-coding:utf-8-*-

from math import acos,pi,sqrt
from decimal import Decimal, getcontext

getcontext().prec = 30

class Vector(object):

    CANNOT_NORMALIZE_ZERO_VECTOR_MSG = 'Cannot compute an angle with the zero vector'

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

print '\ntest...\n'


print 'first pair...'
v = Vector(['-7.579', '-7.88'])
w = Vector(['22.737', '23.64'])
print 'is parallel:', v.is_parallel_to(w)
print 'is orthogonal:', v.is_orthogonal_to(w)

print 'second pair...'
v = Vector(['-2.029', '9.97', '4.172'])
w = Vector(['-9.231', '-6.639', '-7.245'])
print 'is parelled:', v.is_parallel_to(w)
print 'is orthogonal:', v.is_orthogonal_to(w)

print 'third pair...'
v = Vector(['-2.328', '-7.284', '-1.214'])
w = Vector(['-1.821', '1.072', '-2.94'])
print 'is parallel:', v.is_parallel_to(w)
print 'is orthogonal:', v.is_orthogonal_to(w)

print 'fourth pair...'
v = Vector(['2.118', '4.827'])
w = Vector(['0', '0'])
print 'is parellel:', v.is_parallel_to(w)
print 'is orthogonal:', v.is_orthogonal_to(w)
