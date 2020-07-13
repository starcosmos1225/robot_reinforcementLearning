import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import threading
import tensorflow as tf
import numpy as np
import gym
import sys
import shutil
import matplotlib.pyplot as plt


GAME = 'puckworld-v3'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = 4
MAX_GLOBAL_EP = 50000
GLOBAL_NET_SCOPE = 'GLOBAL'
UPDATE_GLOBAL_ITER = 100
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
GLOBAL_HIT = 0
STEP = 1500 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S-1], 'S')
                self.image = tf.placeholder(tf.float32, [None, 20, 20, 1], 'IMAGE')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
                self.saver = tf.train.Saver()
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S-1], 'S')
                self.image = tf.placeholder(tf.float32, [None, 20, 20, 1], 'IMAGE')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))
                with tf.name_scope('a_loss'):

                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            image_l1 = tf.layers.conv2d(inputs=self.image, filters=6, kernel_size=5, strides=(1, 1), padding = 'valid',
                                        activation=tf.nn.relu6, kernel_initializer=w_init)
            image_p1 = tf.layers.max_pooling2d(inputs=image_l1, pool_size=(2, 2), padding='valid',strides=(2,2))
            image_l2 = tf.layers.conv2d(inputs = image_p1, filters=36, kernel_size = 5,strides=(1, 1), padding = 'valid',
                                        activation=tf.nn.relu6, kernel_initializer=w_init)
            image_p2 = tf.layers.max_pooling2d(inputs=image_l2, pool_size=(2, 2), padding='valid', strides=(2, 2))
            image_f2 = tf.layers.flatten(image_p2)
            s1 = tf.concat([self.s, image_f2], axis=1)
            l_a = tf.layers.dense(s1, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            image_l1 = tf.layers.conv2d(inputs=self.image, filters=6, kernel_size=5, strides=(1, 1), padding='valid',
                                        activation=tf.nn.relu6, kernel_initializer=w_init)
            image_p1 = tf.layers.max_pooling2d(inputs=image_l1, pool_size=(2, 2), padding='valid', strides=(2,2))
            image_l2 = tf.layers.conv2d(inputs=image_p1, filters=36, kernel_size=5, strides=(1, 1), padding='valid',
                                        activation=tf.nn.relu6, kernel_initializer=w_init)
            image_p2 = tf.layers.max_pooling2d(inputs=image_l2, pool_size=(2, 2), padding='valid', strides=(2, 2))
            image_f2 = tf.layers.flatten(image_p2)
            s1 = tf.concat([self.s, image_f2], axis=1)
            l_c = tf.layers.dense(s1, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :-1],self.image:s[-1][np.newaxis,:]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action
    def save(self,filename="/A3C_model"):
        path = os.path.split(os.path.realpath(__file__))[0]
        path += filename
        self.saver.save(SESS, path)
    def load(self):
        checkpoint = os.path.split(os.path.realpath(__file__))[0]
        ckpt = tf.train.latest_checkpoint(checkpoint)
        path = checkpoint + "/checkpoint"
        if not os.path.exists(path):
            return False
        # saver = tf.train.Saver()
        #variables = tf.contrib.framework.get_variables_to_restore()

        #banlist = ['la','ap','lc','v']
        #variables_to_resotre = [v for v in variables if v.name.split('/')[0] == 'GLOBAL'
                                #and v.name.split('/')[2] not in banlist]
        #for name in variables_to_resotre:
            #print(name)
        #t=input()
        #self.saver = tf.train.Saver(variables_to_resotre)
        self.saver.restore(SESS, ckpt)
        return True

class Worker(object):
    def __init__(self, name, globalAC,preTrain=False):
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC)
        self.preTrain=preTrain

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP, GLOBAL_HIT
        total_step = 1
        buffer_s,buffer_image, buffer_a, buffer_r = [],[], [], []
        if self.preTrain:
            self.AC.pull_global()
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                #if self.name == 'W_0':
                    #self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                if r < -999 and done:
                    GLOBAL_HIT += 1
                #print("action:{}".format(a))
                #if done: r = -5
                ep_r += r
                buffer_s.append(s[:-1])
                buffer_image.append(s[-1])
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :-1], self.AC.image:s_[-1][np.newaxis,:]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.image:buffer_image,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s,buffer_image,buffer_a, buffer_r = [],[], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done or total_step % 1500==0:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        #GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                        GLOBAL_RUNNING_R.append(ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %f" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params

        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC, False))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())
    GLOBAL_AC.load()
    #t=input("afterload")
    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    testWorker = Worker("test", GLOBAL_AC)
    testWorker.AC.pull_global()

    total_reward = 0
    GLOBAL_AC.save()
    print("the hit times = {}".format(GLOBAL_HIT))
    t=input("begin test...")
    for i in range(TEST):
        state = env.reset()
        for j in range(STEP):
            env.render()
            action = testWorker.AC.choose_action(state)  # direct action for test
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
    ave_reward = total_reward / TEST
    print('episode: ', GLOBAL_EP, 'Evaluation Average Reward:', ave_reward)
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()