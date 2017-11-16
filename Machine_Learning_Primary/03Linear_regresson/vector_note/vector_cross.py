#-*-coding:utf-8-*-

from math import acos,pi,sqrt
from decimal import Decimal, getcontext

getcontext().prec = 30

class Vector(object):

    CANNOT_NORMALIZE_ZERO_VECTOR_MSG = 'Cannot compute an angle with the zero vector'
    NO_UNIQUE_PARALLEL_COMPONENT_MSG = 'NO_UNIQUE_PARALLEL_COMPONENT_MSG'
    NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG = 'NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG'
    ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG = 'ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG'

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
#        coordinates_as_floats = map(float, self.coordinates)
        return 'Vector: {}'.format(self.coordinates)


    def magnitude(self):
        coordinates_squard =[x**Decimal('2.0') for x in self.coordinates]
        return Decimal.sqrt(sum(coordinates_squard))


#cross

    def area_of_triangle_with(self, v):
        return self.area_of_parallelogram_with(v) / Decimal('2.0')

    def area_of_parallelogram_with(self, v):
        cross_product = self.cross(v)
        return cross_product.magnitude()

    def cross(self, v):
        try:
            x_1, y_1, z_1 = self.coordinates
            x_2, y_2, z_2 = v.coordinates
            new_coordinates = [   y_1*z_2 - y_2*z_1 ,
                                -(x_1*z_2 - x_2*z_1),
                                  x_1*y_2 - x_2*y_1   ]
            return Vector(new_coordinates)

        except ValueError as e:
            msg = str(e)
            if msg == 'need more than 2 value to unpack':
                self_embedded_in_R3 = Vector(self.coordinates + ('0',))
                v_embedded_in_R3 = Vector(v.coordinates + ('0',))
                return self_embedded_in_R3.cross(v_embedded_in_R3)
            elif (msg == 'too many values to unpack' or
                  msg == 'need more than 1 value to unpack'):
                raise Exception(self.ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG)
            else:
                raise e


#test
v = Vector(['8.462', '7.893', '-8.187'])
w = Vector(['6.984', '-5.975', '4.778'])
print '#1', v.cross(w)

v = Vector(['-8.987', '-9.838', '5.031'])
w = Vector(['-4.268', '-1.861', '-8.866'])
print '#2', v.area_of_parallelogram_with(w)

v = Vector(['1.5', '9.547', '3.691'])
w = Vector(['-6.007', '0.124', '5.772'])
print '#3', v.area_of_triangle_with(w)

print '\n\n'

v = Vector([-9.88, -3.264, -8.159])
w = Vector([-2.155, -9.353, -9.473])
print 'test,the different between foalt and decimal: ', '\n',v.cross(w)


