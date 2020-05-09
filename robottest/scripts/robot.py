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
from genRandomWorld import genRandomWorld, genEmptyWorld
from fileUpLoad import fileUpLoad, fileDownLoad
import subprocess
import signal
import os
import sys
np.random.seed(2)
tf.set_random_seed(2)  # reproducible

MAX_EPISODE = 3
DISPLAY_REWARD_THRESHOLD = 200
MAX_EP_STEPS = 500
RENDER = False
GAMMA = 0.9
LR_A = 0.001  # Actor
LR_C = 0.01  # Critic
ENTROPY_BETA = 0.001
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
HOST_IP = "192.168.1.102"
USER_NAME= "zhang"
PASSWORD = "Wozuike2"
REMOTE_PATH = "E:/tool"
class ACNet(object):
    def __init__(self, sess, n_features, n_actions, load=False):
        self.sess = sess
        with tf.variable_scope("global"):
            self.s = tf.placeholder(tf.float32, [None, n_features], 'S')
            #self.image = tf.placeholder(tf.float32, [None, 11, 11,1], 'IMAGE')
            self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
            self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
            #self.a_params, self.c_params = \
            self._build_net(n_actions, load)
            td = tf.subtract(self.v_target, self.v, name='TD_error')
            with tf.name_scope('c_loss'):
                self.c_loss = tf.train.RMSPropOptimizer(LR_C).minimize(tf.square(td))
            with tf.name_scope('a_loss'):
                log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, n_actions, dtype=tf.float32), axis=1,
                                        keep_dims=True)
                exp_v = log_prob * tf.stop_gradient(td)
                entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                        axis=1, keep_dims=True)  # encourage exploration
                self.exp_v = tf.reduce_mean(-ENTROPY_BETA * entropy - exp_v)
                self.a_loss = tf.train.RMSPropOptimizer(LR_A).minimize(self.exp_v)

    def _build_net(self,n_actions,load=False):

        with tf.variable_scope('actor'):
            #image_l1 = tf.layers.conv2d(inputs = self.image, filters=6, kernel_size = 4,strides=(1,1),padding = 'valid',
                                        #activation=tf.nn.relu6, kernel_initializer=w_init)
            #image_p1 = tf.layers.max_pooling2d(inputs = image_l1, pool_size = (2,2),padding = 'valid',strides=(2,2))
            #image_l2 = tf.layers.conv2d(inputs = image_p1, filters=6,kernel_size = 4,strides=(1,1),padding = 'valid',
                                        #activation=tf.nn.relu6, kernel_initializer=w_init)
            #image_p2 = tf.layers.flatten(image_l2)
            #s1 = tf.concat([self.s,image_p2],axis=1)
            if load==False:
                w_init = tf.random_normal_initializer(0., .1)
                self.l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
                self.a_prob = tf.layers.dense(self.l_a, n_actions, tf.nn.softmax, kernel_initializer=w_init, name='ap')
            else:
                w_init = tf.random_normal_initializer(0., .1)
                self.l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, name='la')
                self.a_prob = tf.layers.dense(self.l_a, n_actions, tf.nn.softmax, name='ap')
        with tf.variable_scope('critic'):
            #image_l1 = tf.layers.conv2d(inputs=self.image, filters=6, kernel_size=4, strides=(1, 1), padding='valid',
                                        #activation=tf.nn.relu6, kernel_initializer=w_init)
            #image_p1 = tf.layers.max_pooling2d(inputs=image_l1, pool_size=(2, 2), padding='valid',strides=(2,2))
            #image_l2 = tf.layers.conv2d(inputs=image_p1, filters=6, kernel_size=4, strides=(1, 1), padding='valid',
                                        #activation=tf.nn.relu6, kernel_initializer=w_init)
            #image_p2 = tf.layers.flatten(image_l2)
            #s1 = tf.concat([self.s, image_p2], axis=1)
            if load==False:
                self.l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                self.v = tf.layers.dense(self.l_c, 1, kernel_initializer=w_init, name='v')  # state value
            else:
                self.l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, name='lc')
                self.v = tf.layers.dense(self.l_c, 1, name='v')
        #a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        #c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return
    def learn(self, feed_dict):  # run by a local
        self.sess.run([self.a_loss, self.c_loss], feed_dict)  # local grads applies to global net
    def choose_action(self, s):  # run by a local
        prob_weights = self.sess.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action
    def updateNet(self,graph):
        pass
        #self.l_a.weights = graph.get_tensor_by_name("global/actor/la/kernel:0")
        #self.l_a.bias = graph.get_tensor_by_name("global/actor/la/bias:0")
        #self.a_prob.weights = graph.get_tensor_by_name("global/actor/ap/kernel:0")
        #self.a_prob.bias = graph.get_tensor_by_name("global/actor/ap/bias:0")
        #self.l_c.weights = graph.get_tensor_by_name("global/critic/lc/kernel:0")
        #self.l_c.bias = graph.get_tensor_by_name("global/critic/lc/bias:0")
        #self.v.weights = graph.get_tensor_by_name("global/critic/v/kernel:0")
        #self.v.bias = graph.get_tensor_by_name("global/critic/v/bias:0")

class Robot:
    def __init__(self, env):
        self.iterator_n = 0
        self.action_n = 4
        self.env = env
        self.state_n = 4
        self.sess = tf.Session()
        #checkpoint = os.path.split(os.path.realpath(__file__))[0]
        #path = checkpoint + "/model.ckpt.meta"
        #if not os.path.exists(path):
        self.ACNet = ACNet(self.sess, n_features=self.state_n, n_actions=self.action_n)
        self.sess.run(tf.global_variables_initializer())
        #else:
            #self.ACNet = ACNet(self.sess, n_features=self.state_n, n_actions=self.action_n,load=True)  # AC
            #self.sess.run(tf.global_variables_initializer())
        self.load()
        #variable_names = [v.name for v in tf.trainable_variables()]
        #print(variable_names)
        #t = input()
        self.child = None
        ep = 1.0
        reward_list = []
        avg_reward = 0.0
    def initEnv(self):
        # roslaunch husky_gazebo husky_RL.launch x:=<location X> y:=<location Y>
        global receiveGoal
        path = os.path.split(os.path.realpath(__file__))[0]
        #path += "/../worlds/random.world"
        path += "/../worlds/empty.world"
        #x, y, goal.x, goal.y= genRandomWorld(path, 0)
        x, y, goal.x, goal.y = genEmptyWorld(path)

        receiveGoal = True
        locationX = "x:={}".format(x)
        locationY = "y:={}".format(y)
        rospy.sleep(2)
        self.child = subprocess.Popen(["roslaunch", "robottest", "husky_RL.launch", locationX, locationY,"gui:=true"])
        rospy.sleep(4)
        print("location:{} {}".format(x, y))
        print("goal:{} {}".format(goal.x, goal.y))
        pass
    def train(self):
        for i_episode in range(MAX_EPISODE):
            self.initEnv()
            total_step = 0
            print("begin learn:{}".format(i_episode))
            #rospy.sleep(10)
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            while True:
                s = self.env.getState()
                a = self.ACNet.choose_action(s)  # Actor
                self.env.step(a)
                rospy.sleep(0.2)
                s_, r, done, info = self.env.getStatus()
                ep_r += r
                buffer_s.append(s)
                #buffer_image.append(s[-1])
                buffer_a.append(a)
                buffer_r.append(r)
                total_step += 1
                print("step:{} action:{} distance:{}".format(total_step, a, s_[0]))
                if total_step % MAX_EP_STEPS == 0 or done:
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.sess.run(self.ACNet.v, {self.ACNet.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.ACNet.s: buffer_s,
                        #self.AC.image:buffer_image,
                        self.ACNet.a_his: buffer_a,
                        self.ACNet.v_target: buffer_v_target,
                    }
                    self.update(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    print(
                        "Ep:", i_episode,
                        "| Ep_r: %f" % ep_r
                    )
                    break
                #if done or total_step % MAX_EP_STEPS == 0:
                    # if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                    # else:
                    # GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    #print(
                        #"Ep:", i_episode,
                        #"| Ep_r: %f" % ep_r
                    #)
                    #break
            if self.child is not None:
                #subprocess.Popen(["rosservice", "call", "/gazebo/delete_model", "model_name: '/'"])
                #rospy.sleep(0.5)
                self.child.send_signal(signal.SIGINT)
                rospy.sleep(2)

    def save(self, filename="/model.ckpt"):
        saver = tf.train.Saver()
        path = os.path.split(os.path.realpath(__file__))[0]
        path += filename
        saver.save(self.sess, path)
        pass
    def load(self):
        checkpoint = os.path.split(os.path.realpath(__file__))[0]
        path = checkpoint + "/model.ckpt"
        if not os.path.exists(path):
            return False
        #saver = tf.train.import_meta_graph(path)
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        #t=input()
        #self.graph = tf.get_default_graph()
        return True
    def update(self,data):
        # first download model from remote computer.
        # Then like load(),read from model and read the graph
        # Then renew the net. But we need train the model with storage data
        # then save the net as model_upload.*** and call upload() to transport the file to remote computer
        #######################################################
        # first download model
        self.download()
        # Then like load(),read from model and read the graph
        self.load()
        #self.ACNet.updateNet(self.graph)# new version will be ACNET. It merge the actor and critic
        # Then renew the net. But we need train the model with storage data
        self.ACNet.learn(data)
        # then save the net as model_upload.*** and call upload() to transport the file to remote compu
        self.save()
        self.upload()

        pass
    def download(self):
        fileDownLoad(filename='/model.ckpt.meta', host_ip=HOST_IP, username=USER_NAME,
                     password=PASSWORD, remotepath=REMOTE_PATH+'/model.ckpt.meta')
        fileDownLoad(filename='/model.ckpt.index', host_ip=HOST_IP, username=USER_NAME,
                     password=PASSWORD, remotepath=REMOTE_PATH+'/model.ckpt.index')
        fileDownLoad(filename='/model.ckpt.data-00000-of-00001', host_ip=HOST_IP, username=USER_NAME,
                     password=PASSWORD, remotepath=REMOTE_PATH+'/model.ckpt.data-00000-of-00001')
        fileDownLoad(filename='/checkpoint', host_ip=HOST_IP, username=USER_NAME,
                     password=PASSWORD, remotepath=REMOTE_PATH+'/checkpoint')
    def upload(self):
        # upload the model
        # the name is /model_upload.***
        #######################################################
        fileUpLoad(filename='/model.ckpt.meta', host_ip=HOST_IP, username=USER_NAME,
                     password=PASSWORD, remotepath=REMOTE_PATH + '/model.ckpt.meta')
        fileUpLoad(filename='/model.ckpt.index', host_ip=HOST_IP, username=USER_NAME,
                     password=PASSWORD, remotepath=REMOTE_PATH + '/model.ckpt.index')
        fileUpLoad(filename='/model.ckpt.data-00000-of-00001', host_ip=HOST_IP, username=USER_NAME,
                     password=PASSWORD, remotepath=REMOTE_PATH + '/model.ckpt.data-00000-of-00001')
        #fileUpLoad(filename='/checkpoint', host_ip=HOST_IP, username=USER_NAME,
                     #password=PASSWORD, remotepath=REMOTE_PATH + '/checkpoint')
        pass

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
def test():
    robot = Robot(env)

    global receiveGoal
    path = os.path.split(os.path.realpath(__file__))[0]
    # path += "/../worlds/random.world"
    path += "/../worlds/empty.world"
    # x, y, goal.x, goal.y= genRandomWorld(path, 0)
    x, y, goal.x, goal.y = genEmptyWorld(path)
    receiveGoal = True
    locationX = "x:={}".format(x)
    locationY = "y:={}".format(y)
    rospy.sleep(2)
    subprocess.Popen(["roslaunch", "robottest", "husky_RL.launch",locationX,locationY])
def init():
    global pubAction, pubStopMovebase
    rospy.init_node('mainController', anonymous=True)
    pubAction = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    env.setPublish(pubAction)
    pubStopMovebase = rospy.Publisher('/move_base/cancel', GoalID, queue_size = 10)
    subPosition = rospy.Subscriber("/odometry/filtered", Odometry, getPosition)
    subGoal = rospy.Subscriber("/move_base_simple/goal", PoseStamped, getGoal)
    robot = Robot(env)
    try:
        robot.download()
    except Exception as e:
        print(e.args)
        print("download fail")
    robot.train()
    print("end train")
    #robot.save()
    #print("end save")
    rospy.spin()

if __name__=='__main__':
    try:

        # child.terminate()
        init()
    except rospy.ROSInterruptException as e:
        print(e.args[0])