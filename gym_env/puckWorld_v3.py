import math
import matplotlib.pyplot as plt
import random
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from PIL import Image

class vec2d:
    def __init__(self, x_=0.0, y_=0.0):
        self.x = x_
        self.y = y_

    def __mul__(self, v):
        # type:(Vec2d)->Vec2d
        return vec2d(self.x*v.x,self.y*v.y)

    def __mul__(self,v):
        # type:(Vec2d)->float
        return vec2d(self.x*v,self.y*v)

    def __truediv__(self, scale):
        return vec2d(self.x/scale, self.y/scale)
    def cross(self,v):
        return self.x*v.y-self.y*v.x

    def dot(self,v):
        return self.x*v.x+self.y*v.y

    def __add__(self,v):
        return vec2d(self.x+v.x,self.y+v.y)

    def __sub__(self,v):
        return vec2d(self.x-v.x,self.y-v.y)
    def length(self):
        return math.sqrt(self.x**2+self.y**2)
    def norm(self):
        #print("x:{},y:{}".format(self.x, self.y))
        l = self.length()
        #print("x:{},y:{}".format(self.x,self.y))
        #print(l)
        if math.fabs(l-0.0)<1e-16:
            return vec2d(0,0)
        return vec2d(self.x/l, self.y/l)

class puckworldEnv_v3(gym.Env):
    """
    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Car PositionX              0.0          200.0
        1	Car PositionY              0.0          200.0
        2	Car PoseX                  -1            1
        3	Car PoseY                  -1            1
        4   target PositionX           0.0          200.0
        5   target PositionY           0.0          200.0
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	set Velocity 1
        1	set Velocity 0
        2   turn left with rad w
        3   turn right with rad w
    Reward:
        Reward is -1 for every step taken, including the termination step
        If get to target position reward +100

    Starting State:
        set Random car position in 0-20
        set pose random in 0-1
        set target random in x[50-100] Y[20-100]

    Episode Termination:
        the distance between car and target is smaller than 1.0
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.width = 400
        self.height = 400
        self.image = np.zeros((self.width, self.height), dtype='int32')
        self.view_radius = 50
        self.compress = 5
        self.velocity = 0
        self.rad = 0
        self.threshold = 20.0
        self.obstacle_number = 20
        self.obstacle_size = 20
        self.high = np.array([
            599.0,
            2.0*math.pi,
            1.0,
            1.0,
            1.0])
        self.low = np.array([
            0.0,
            0.0,
            -1.0,
            -1.0,
            -1.0])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.reward = 200
        self.obstacle_reward = -10000
        self.seed()
        self.viewer = None
        self.state = None
        self.carSize = 12
        self.steps_beyond_done = None
        self.obstacle_view = [None] * self.obstacle_number
        self.ob_r = [None] * self.obstacle_number
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def move(self):
        car_x, car_y, tar_x, tar_y, pose_x, pose_y = self.state
        location = vec2d(car_x, car_y)
        direction = vec2d(pose_x, pose_y).norm()
        newdirection = vec2d(direction.x*math.cos(self.rad)-direction.y*math.sin(self.rad),
                             direction.y*math.cos(self.rad)+direction.x*math.sin(self.rad))
        newdirection = newdirection.norm()
        newlocation = location+newdirection*self.velocity
        hit = False
        if newlocation.x < 0 or newlocation.x > self.width:
            hit = True
        if newlocation.y < 0 or newlocation.y > self.height:
            hit = True
        newlocation.x = max(min(400, newlocation.x), 0.0)
        newlocation.y = max(min(400, newlocation.y), 0.0)
        self.state = (newlocation.x, newlocation.y, tar_x, tar_y, newdirection.x, newdirection.y)
        return hit
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        #state = self.state
        car_x, car_y, target_X, target_Y, pose_x, pose_y = self.state
        #print("pose2:{} {}".format(pose_x,pose_y))
        old_distance = math.sqrt((target_X - car_x) ** 2 + (target_Y - car_y) ** 2)
        v = self.velocity
        r = self.rad
        if action == 0:
            self.velocity = 4
            self.rad = -0.2
            #car_x -= self.velocity
            #car_x = max(0,car_x)
        elif action == 1:
            self.velocity = 10
            self.rad = 0
            #car_x += self.velocity
            #car_x = min(199.0, car_x)
        elif action == 2:
            self.velocity = 4
            self.rad = 0.2
            #car_y -= self.velocity
            #car_y = max(0, car_y)
        #elif action == 3:
            #self.velocity = 0
            #self.rad = 0
            #car_y += self.velocity
            #car_y = min(199.0, car_y)
        hit = self.move()
        #print("pose3:{} {}".format(pose_x, pose_y))
        #t=input()
        car_x1, car_y1, target_X1, target_Y1, pose_x1, pose_y1 = self.state
        #print("pose1:{} {}".format(pose_x1,pose_y1))
        distance, theta = self.trans2dt(car_x1, car_y1, target_X1, target_Y1, pose_x1, pose_y1)
        self.state_mat = (distance, theta, self.velocity, self.rad, self.getImage())
        done = distance < self.threshold
        if not hit:
            hit = self.in_obstacle(car_x1, car_y1)
        if not done:
            #current_reward = old_distance-distance
            current_reward = 10*(old_distance-0.00001-distance)-0.001*(math.fabs(self.velocity-v)+math.fabs(self.rad-r))
            if hit:
                current_reward = self.obstacle_reward
            #if current_reward <0:
                #current_reward=-1
            #elif current_reward>0:
                #current_reward=1
            #if (old_distance-distance<1e-5):
                #current_reward -= 0.01
        else:
            current_reward = self.reward
            print("reach goal")
            #current_reward = old_distance - distance
            #if current_reward < 0:
                #current_reward = -1
            #elif current_reward > 0:
                #current_reward = 1
        if hit:
            done=True
            print("hit occur")
        return np.array(self.state_mat), current_reward, done, {}

    def trans2mat(self, car_x, car_y, tar_x, tar_y):
        state_mat = np.mat(np.zeros((200, 200)))
        state_mat[int(car_x/1), int(car_y/1)] = 255.0
        state_mat[int(tar_x/1), int(tar_y/1)] = -255.0
        return state_mat

    def trans2dt(self, car_x, car_y, tar_x, tar_y, pose_x, pose_y):
        d = math.sqrt((tar_x-car_x) ** 2+(tar_y-car_y) ** 2)
        direction1 = vec2d(tar_x - car_x, tar_y - car_y).norm()
        direction2 = vec2d(-pose_x, -pose_y).norm()
        sintheta = direction2.cross(direction1)
        costheta = direction1.dot(direction2)
        if sintheta > 0:
            theta = 2*math.pi-math.acos(costheta)
        else:
            theta = math.acos(costheta)
        return d, theta

    def createObstacle(self, n):
        limit = int(self.width*self.height/1600)
        l = random.sample(range(0, limit), n)
        self.obstacle = np.zeros((n, 2), dtype='int32')
        for i in range(n):
            self.obstacle[i, 0] = int(l[i] / 10)
            self.obstacle[i, 1] = l[i] % 10
            self.obstacle[i, 0] = int(self.obstacle[i, 0] * 40)
            self.obstacle[i, 1] = int(self.obstacle[i, 1] * 40)
            for j in range(self.obstacle[i, 0]-int(self.obstacle_size/2), self.obstacle[i, 0] + int(self.obstacle_size/2)):
                for k in [self.obstacle[i, 1]-int(self.obstacle_size/2), self.obstacle[i, 1] + int(self.obstacle_size/2)]:
                    if 0 <= j <= self.width and 0 <= k <= self.height:
                        self.image[j, k] = 1.0
            for j in range(self.obstacle[i, 1]-int(self.obstacle_size/2), self.obstacle[i, 1] + int(self.obstacle_size/2)):
                for k in [self.obstacle[i, 0]-int(self.obstacle_size/2), self.obstacle[i, 0] + int(self.obstacle_size/2)]:
                    if 0 <= k <= self.width and 0 <= j <= self.height:
                        self.image[k, j] = 1.0
        return None

    def getImage(self):
        x, y, tx, ty, pose_x, pose_y = self.state
        loc = vec2d(x, y)
        v_vec = vec2d(pose_y, -pose_x)
        pose = vec2d(pose_x, pose_y)
        lr_loc = loc - pose*self.view_radius - v_vec*self.view_radius
        image = np.zeros((self.view_radius * 2, self.view_radius * 2, 1))
        for i in range(self.view_radius * 2):
            tmp_loc_row = lr_loc+pose*i
            for j in range(self.view_radius * 2):
                tmp_loc = tmp_loc_row + v_vec*j
                x_ = int(tmp_loc.x)
                y_ = int(tmp_loc.y)
                #print("loc now:{} {}".format(x_, y_))
                if x_ < 0 or x_ >= self.width or y_ < 0 or y_ >= self.height:
                    image[i, j, 0] = 1
                    #showimage[self.view_radius*2-1-i, j] = 0
                else:
                    image[i, j, 0] = self.image[x_, y_]
                    #showimage[self.view_radius*2-1-i, j] = 0 if self.image[x_, y_] == 1 else 255
        #print(image)
        #t=input()

        return self.compressMap(image, compression=self.compress)

    def in_obstacle(self, x, y):
        for i in range(int(x)-int(self.carSize/2), int(x)+int(self.carSize/2)):
            for j in range(int(y)-int(self.carSize/2), int(y)+int(self.carSize/2)):
                if i <= 0 or i >= self.width:
                    return True
                if j <= 0 or j >= self.width:
                    return True
                if self.image[i, j] == 1.0:
                    return True
        return False

    def reset(self):
        self.image = np.zeros((self.width, self.height), dtype='int8')
        self.createObstacle(n=self.obstacle_number)
        (car_x,car_y) = self.np_random.uniform(low=0, high=399, size=(2,))
        while self.in_obstacle(car_x, car_y):
            (car_x, car_y) = self.np_random.uniform(low=0, high=399, size=(2,))
        (pose_x, pose_y) = self.np_random.uniform(low=-1.00, high=1.0, size=(2,))
        d=0
        #print("13:{} {}".format(pose_x,pose_y))
        pose = vec2d(pose_x, pose_y).norm()
        (pose_x, pose_y) = (pose.x, pose.y)
        while d < self.threshold:
            (tar_x, tar_y) = self.np_random.uniform(low=0, high=399, size=(2,))
            while self.in_obstacle(tar_x, tar_y):
                (tar_x, tar_y) = self.np_random.uniform(low=0, high=399, size=(2,))
            #print("tarx:{},tary:{}".format(tar_x,tar_y))
            #t=input()
            d =math.sqrt((tar_x-car_x)**2+(tar_y-car_y)**2)
        self.velocity = 0.0
        self.rad = 0.0
        d,theta = self.trans2dt(car_x, car_y, tar_x, tar_y, pose_x, pose_y)
        self.state = (car_x, car_y, tar_x, tar_y, pose_x, pose_y)
        self.state_mat =(d, theta, self.velocity, self.rad, self.getImage())
        self.steps_beyond_done = None
        #self.getImage()
        return np.array(self.state_mat)
    def check(self,state):
        car_x, car_y, target_X, target_Y = self.state
        distance = math.sqrt((target_X - state[0,0]) ** 2 + (target_Y - state[0,1]) ** 2)
        return distance < self.threshold,self.reward
    def render(self, mode='human'):
        screen_width = 400
        screen_height = 400

        car_radius = 4
        target_radius = 4
        car_color = [.9, .0, .0]
        target_color = [.0, .0, .9]
        obstacle_color = [0, 0, 0]
        car_x, car_y, target_x, target_y,pose_x,pose_y = self.state
        n = self.obstacle.shape[0]
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.carTrans = rendering.Transform(translation=(car_x, car_y))
            self.car = rendering.make_circle(car_radius)
            self.car.add_attr(self.carTrans)
            self.car.set_color(car_color[0], car_color[1], car_color[2])
            self.tarTrans = rendering.Transform(translation=(target_x, target_y))
            self.target = rendering.make_circle(target_radius)
            self.target.add_attr(self.tarTrans)
            self.target.set_color(target_color[0], target_color[1], target_color[2])
            self.viewer.add_geom(self.car)
            self.viewer.add_geom(self.target)
            for j in range(n):
                self.obstacle_view[j] = rendering.Transform(translation=(self.obstacle[j, 0], self.obstacle[j, 1]))
                v = [[-self.obstacle_size/2, - self.obstacle_size/2],
                     [-self.obstacle_size/2, self.obstacle_size/2],
                     [self.obstacle_size/2, self.obstacle_size/2],
                     [self.obstacle_size/2, -self.obstacle_size/2]]
                #tt = [[-5,-5],[5,-5],[5,5],[-5,5]]
                self.ob_r[j] = rendering.make_polygon(v, filled=True)
                self.ob_r[j].add_attr(self.obstacle_view[j])
                self.ob_r[j].set_color(obstacle_color[0], obstacle_color[1], obstacle_color[2])
                self.viewer.add_geom(self.ob_r[j])

        if self.state is None: return None
        x = self.state
        self.carTrans.set_translation(x[0], x[1])
        self.tarTrans.set_translation(x[2], x[3])
        for j in range(n):
            self.obstacle_view[j].set_translation(self.obstacle[j, 0], self.obstacle[j, 1])
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def compressMap(self, map_image, compression=1):
        if compression == 1:
            return map_image.copy()
        size = map_image.shape[0]
        s = int(size/compression)
        data = np.zeros((s, s, 1))
        for i in range(size):
            for j in range(size):
                x = int(i/compression)
                y = int(j/compression)
                if map_image[i, j, 0] > 1e-6:
                    data[x, y, 0] = 1.0
        return data.copy()