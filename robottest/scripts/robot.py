#! /usr/bin/env python
import rospy
import numpy as np
import tensorflow as tf
import random
import math
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from actionlib_msgs.msg import GoalID
from nav_msgs.msg import Odometry
from Env import Interact
from Vec import Vec2d
from genRandomWorld import genRandomWorld
import subprocess
import signal
import os
import sys
np.random.seed(2)
tf.set_random_seed(2)  # reproducible

MAX_EPISODE = 20
DISPLAY_REWARD_THRESHOLD = 200
MAX_EP_STEPS = 500
RENDER = False
GAMMA = 0.9
LR_A = 0.001  # Actor
LR_C = 0.01  # Critic

N_F = 4#env.observation_space.shape[0]
N_A = 4#env.action_space.n

pubAction = None
pubStopMovebase = None
env = Interact(4)
goal = Vec2d()
position = Vec2d()
pose = Vec2d()
initPose = np.mat([[1, 0, 0]])
receiveGoal = False

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001, graph=None):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            if graph is None:
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=20,  # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='l1'
                )
                l2 = tf.layers.dense(
                    inputs=l1,
                    units=20,  # number of hidden units
                    activation=tf.nn.tanh,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='l2'
                )
                self.acts_prob = tf.layers.dense(
                    inputs=l2,
                    units=n_actions,  # output units
                    activation=tf.nn.softmax,  # get action probabilities
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='acts_prob'
                )
            else:
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=20,  # number of hidden units
                    activation=tf.nn.relu,
                    #kernel_initializer=graph.get_tensor_by_name("Actor/l1/kernel:0"),  # weights
                    #bias_initializer=graph.get_tensor_by_name("Actor/l1/bias:0"),  # biases
                    name='l1'
                )
                l1.weights = graph.get_tensor_by_name("Actor/l1/kernel:0")
                l1.bias = graph.get_tensor_by_name("Actor/l1/bias:0")
                l2 = tf.layers.dense(
                    inputs=l1,
                    units=20,  # number of hidden units
                    activation=tf.nn.tanh,
                    #kernel_initializer=graph.get_tensor_by_name("Actor/l2/kernel:0"),  # weights
                    #bias_initializer=graph.get_tensor_by_name("Actor/l2/bias:0"),  # biases
                    name='l2'
                )
                l2.weights = graph.get_tensor_by_name("Actor/l2/kernel:0")
                l2.bias = graph.get_tensor_by_name("Actor/l2/bias:0")
                self.acts_prob = tf.layers.dense(
                    inputs=l2,
                    units=n_actions,  # output units
                    activation=tf.nn.softmax,  # get action probabilities
                    #kernel_initializer=graph.get_tensor_by_name("Actor/acts_prob/kernel:0"),  # weights
                    #bias_initializer=graph.get_tensor_by_name("Actor/acts_prob/bias:0"),  # biases
                    name='acts_prob'
                )
                self.acts_prob.weights = graph.get_tensor_by_name("Actor/acts_prob/kernel:0")
                self.acts_prob.bias = graph.get_tensor_by_name("Actor/acts_prob/bias:0")
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01, graph=None):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            if graph is None:
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=20,  # number of hidden units
                    activation=tf.nn.relu,  # None
                    # have to be linear to make sure the convergence of actor.
                    # But linear approximator seems hardly learns the correct Q.
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='l1'
                )
                l2 = tf.layers.dense(
                    inputs=l1,
                    units=20,  # number of hidden units
                    activation=tf.nn.tanh,  # None
                    # have to be linear to make sure the convergence of actor.
                    # But linear approximator seems hardly learns the correct Q.
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='l2'
                )
                self.v = tf.layers.dense(
                    inputs=l2,
                    units=1,  # output units
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='V'
                )
            else:
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=20,  # number of hidden units
                    activation=tf.nn.relu,  # None
                    # have to be linear to make sure the convergence of actor.
                    # But linear approximator seems hardly learns the correct Q.
                    #kernel_initializer=graph.get_tensor_by_name("Critic/l1/kernel:0"),  # weights
                    #bias_initializer=graph.get_tensor_by_name("Critic/l1/bias:0"),  # biases
                    name='l1'
                )
                l1.weights = graph.get_tensor_by_name("Critic/l1/kernel:0")
                l1.bias = graph.get_tensor_by_name("Critic/l1/bias:0")
                l2 = tf.layers.dense(
                    inputs=l1,
                    units=20,  # number of hidden units
                    activation=tf.nn.tanh,  # None
                    # have to be linear to make sure the convergence of actor.
                    # But linear approximator seems hardly learns the correct Q.
                    #kernel_initializer=graph.get_tensor_by_name("Critic/l2/kernel:0"),  # weights
                    #bias_initializer=graph.get_tensor_by_name("Critic/l2/bias:0"),  # biases
                    name='l2'
                )
                l2.weights = graph.get_tensor_by_name("Critic/l2/kernel:0")
                l2.bias = graph.get_tensor_by_name("Critic/l2/bias:0")
                self.v = tf.layers.dense(
                    inputs=l2,
                    units=1,  # output units
                    activation=None,
                    #kernel_initializer=graph.get_tensor_by_name("Critic/V/kernel:0"),  # weights
                    #bias_initializer=graph.get_tensor_by_name("Critic/V/bias:0"),  # biases
                    name='V'
                )
                self.v.weights = graph.get_tensor_by_name("Critic/V/kernel:0")
                self.v.bias = graph.get_tensor_by_name("Critic/V/bias:0")
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error

class Robot:
    def __init__(self, env):
        self.sess = tf.Session()
        if not self.load():
            self.actor = Actor(self.sess, n_features=N_F, n_actions=N_A, lr=LR_A)  # Actor
            self.critic = Critic(self.sess, n_features=N_F, lr=LR_C)  # Critic
        else:
            print("load")
            self.actor = Actor(self.sess, n_features=N_F, n_actions=N_A, lr=LR_A, graph=self.graph)  # Actor
            self.critic = Critic(self.sess, n_features=N_F, lr=LR_C, graph=self.graph)  # Critic
        self.sess.run(tf.global_variables_initializer())
        self.iterator_n = 0
        self.action_n = 4
        self.env = env
        self.child = None
        ep = 1.0
        reward_list = []
        avg_reward = 0.0
    def initEnv(self):
        # roslaunch husky_gazebo husky_RL.launch x:=<location X> y:=<location Y>
        global receiveGoal
        path = os.path.split(os.path.realpath(__file__))[0]
        path += "/../worlds/random.world"
        x, y, goal.x, goal.y= genRandomWorld(path, 0)
        #print("init location:{} {}".format(x, y))
        #print("init goal:{} {}".format(goal.x, goal.y))
        receiveGoal = True
        locationX = "x:={}".format(x)
        locationY = "y:={}".format(y)
        #x, y = self.env.getRobotLocation()
        locationX += str(x)
        locationY += str(y)
        self.child = subprocess.Popen(["roslaunch", "robottest", "RL_Env.launch", locationX, locationY])
        rospy.sleep(5)
        pass
    def train(self):
        for i_episode in range(MAX_EPISODE):
            self.initEnv()
            t = 0
            print("begin learn:{}".format(i_episode))
            #rospy.sleep(30)
            track_r = []
            while True:
                s = self.env.getState()
                a = self.actor.choose_action(s)  # Actor
                self.env.step(a)
                rospy.sleep(0.1)
                s_, r, done, info = self.env.getStatus()
                track_r.append(r)
                td_error = self.critic.learn(s, r, s_)  # Critic
                self.actor.learn(s, a, td_error)  # Actor
                t += 1
                #print("step:{}".format(t))
                if done or t >= MAX_EP_STEPS:
                    ep_rs_sum = sum(track_r)
                    running_reward = ep_rs_sum
                    print("episode:", i_episode, "  reward:", running_reward)
                    break
            if self.child is not None:
                self.child.send_signal(signal.SIGINT)
                rospy.sleep(2)
    def save(self):
        saver = tf.train.Saver()
        path = os.path.split(os.path.realpath(__file__))[0]
        path += "/model.ckpt"
        saver.save(self.sess, path)
        pass
    def load(self):
        checkpoint = os.path.split(os.path.realpath(__file__))[0]
        path = checkpoint + "/model.ckpt.meta"
        #print(path)
        if not os.path.exists(path):
            return False
        saver = tf.train.import_meta_graph(path)
        #t=input("t")
        saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint))
        #t=input("tt")
        self.graph = tf.get_default_graph()
        #variable_names = [v.name for v in tf.trainable_variables()]
        #print(variable_names)
        #t=input("ttt")
        return True

def getPosition(data):
    #print("receive position:{} {}".format(data.pose.pose.position.x,data.pose.pose.position.y))
    position.x = data.pose.pose.position.x
    position.y = data.pose.pose.position.y
    x, y, z, w = data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w
    rotateMat = np.mat([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*w*y+2*x*z],
                        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                        [2*x*z-2*w*y, 2*w*x+2*y*z, 1-2*x*x-2*y*y]])
    pose_ = initPose*rotateMat
    pose2d = Vec2d(pose_[0,0], pose_[0,1]).norm()
    pose.x = pose2d.x
    pose.y = pose2d.y
    if receiveGoal:
        d = goal-pose
        distance = d.length()
        d = d.norm()
        sintheta = pose.cross(d)
        costheta = d.dot(pose)
        if sintheta > 0:
            theta = 2 * math.pi - math.acos(costheta)
        else:
            theta = math.acos(costheta)
        env.setState(distance, 0)
        env.setState(theta, 1)
def setCollusion(data):
    env.setCollusion(True)
def getGoal(data):
    #print("get goal!")
    goal.x = data.pose.position.x
    goal.y = data.pose.position.y
    msg = GoalID()
    pubStopMovebase.publish(msg)
    receiveGoal = True

def init():
    global pubAction,pubStopMovebase
    rospy.init_node('mainController', anonymous=True)
    pubAction = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    env.setPublish(pubAction)
    pubStopMovebase = rospy.Publisher('/move_base/cancel',GoalID,queue_size = 10)
    subPosition = rospy.Subscriber("/odometry/filtered", Odometry, getPosition)
    subGoal = rospy.Subscriber("/move_base_simple/goal", PoseStamped, getGoal)
    robot = Robot(env)
    robot.train()
    print("end train")
    robot.save()
    print("end save")
    rospy.spin()

if __name__=='__main__':
    try:

        # child.terminate()
        init()
    except rospy.ROSInterruptException as e:
        print(e.args[0])