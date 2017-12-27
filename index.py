import tensorflow as tf
from math import pi
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

def BuildGraph(radius):

    radius = tf.Variable(radius)
    radius.initializer.run()
    pi_tensor = tf.constant(pi)
    diameter_tensor = tf.multiply(radius, tf.constant(2.0))
    circle_circumference_graph = tf.multiply(diameter_tensor, pi_tensor)

    square_powered = tf.pow(radius, 2)
    circle_area_graph = tf.multiply(pi_tensor, square_powered)

    return (circle_area_graph, circle_circumference_graph, radius)


radius = 0.0
session = tf.InteractiveSession()
area, circumference, radius_tensor = BuildGraph(radius)

while True:

    radius_str = input('Enter the radius of cirlce?')
    radius = float(radius_str)
    tf.assign(radius_tensor, radius).eval()
    circumferece_value = circumference.eval()
    area_value = area.eval()

    print('Circumference value: ' + str(circumferece_value))
    print('Area value: ' + str(area_value))

    takeNext = input('Another time? ')
    if takeNext == 'n' or takeNext == 'N':
        break


session.close()