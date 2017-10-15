# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5,3,0.9) #5 corressponds to the states encoded vectors of input, 3 - possible actions - go left, right or straight. 
#0.9 is again the parameter in the deque learning algorithm.
action2rotation = [0,20,-20] # vectro of 3 elements - actions are encoded by 3 numbers. 
#If 0(index of action) is 0 - corresponds to going left, If 1 then straight. The code will go 20 degrees
#to the specified direction. The code will go -20 degrees and go to the left.         
last_reward = 0 # If car doesn't go into sand it'll be positive or else it'll be negative. 
scores = [] # scores - vector that contains the rewards so that you can make a curve of the mear square 
#reward with respect to time.  

# Initializing the map
first_update = True
def init():
    global sand # array in which cells will be the pixels of the map    
    global goal_x # the destination - upper left corner of the map.
    global goal_y # 
    global first_update
    sand = np.zeros((longueur,largeur))
    goal_x = 20 # goal is not 0 because it should not touch the wall 
    goal_y = largeur - 20 # 
    first_update = False

# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0) # angle between the x and y axis
    rotation = NumericProperty(0) # rotation of the last rotation which is either 0, 20 or -20
    velocity_x = NumericProperty(0) # x - coordinate of velocity vector
    velocity_y = NumericProperty(0)# y - coordinate of the velocity vector
    velocity = ReferenceListProperty(velocity_x, velocity_y) 
    sensor1_x = NumericProperty(0) # Will detect if anything is there straight of the car
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0) # Will detect if anything is there left of the car
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0) # Will detect if anything is there right of the car
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    #From the sensor we get the signals - signal1 corres to sensor1 and so on. 
    #How to compute - Take big squares around each of the sensors, 200 * 200 cells - we divide the number of 1s
    # in the square with the total number of 1s in the square.
    signal1 = NumericProperty(0)  
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)
    
    #function that will make the car move to left, right and straight.
    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos #updating the position - in the direction of the velocity vector.
        self.rotation = rotation #How to rotate the car going to the left or the right
        self.angle = self.angle + self.rotation # angle between the x-axis and the direction of the car
        #Update the sensors and the signals - using the rotate function to which we add the new position
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos 
        #why 30 - it is the distance between the car and the sensor
        #the distance between the car and what the car detects.
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        #update the signals - given the x-coordinate of our sensors then we take all the cells from -10 to 10
        #same for y-coordinates, therefore we get a square of 20 by 20 pixels surrounding the sensor we sum all the ones
        #and divide by 400 to get the density of ones inside the square - thats how we get the signal        
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        
        #punishing the car if it goes near the wall
        # self.sensor1_x>longueur-10 - car getting close to right
        # self.sensor1_x<10 - car getting close to left
        # self.sensor1_y>largeur-10 - car is getting closer to the upper right
        # self.sensor1_y<10 - car is getting closer to the upper left
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.


# So far we have created the car, now we'll create the map - the game it self.
            
class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class - Game is to go from airport to downtown

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    #The action the car has to do to accomplish its goal 
    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180. #If the car is heading towards the goal the orientation will be 0.
        #Slightly to the right orientation will be close to 45. Left will be -45 degrees.
        # -orientation doesn't really matter. Because the NN will fix this with the weights. 
        # - adding this stabilizes the exploration - AI doesn't always explore in the same direction
        #this is the input to our NN. Ouput is the action to be played.  
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation] 
        action = brain.update(last_reward, last_signal) #output of our neural network - heart of our AI
        scores.append(brain.score()) # update the scores 
        rotation = action2rotation[action] # you will select the action and get the rotation
        self.car.move(rotation) #moves according to the action that was selected
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2) 
        # we take the distance of the car to the road and we will get the positions of the sensor ball 
        # one ball two and three   
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        #This is where we penalize the car if it goes into some sand - slow down - v from 6 to 1
        #Gets bad reward.
        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else: # otherwise keeps its speed - if going towards the goal then positive reward. Else negative reward.
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2  
            if distance < last_distance:
                last_reward = 0.1
        
        # Punishing the car for getting too close to walls         
        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1
        
        #update the goal once it reaches the goal.
        if distance < 100:
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y
        last_distance = distance

# Adding the painting tools
#This is the map for commented version check resources
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save() #Saving the brain so that we can reuse it later by taking the load function below 
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load() # load the memory of the core 

# Running the whole thing
if __name__ == '__main__':
    CarApp().run() #Runs the map and the AI itself.
