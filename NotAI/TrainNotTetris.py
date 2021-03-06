"""
    TensorFlow translation of the torch example found here (written by SeanNaren).
    https://github.com/SeanNaren/TorchQLearningExample

    Original keras example found here (written by Eder Santana).
    https://gist.github.com/EderSantana/c7222daa328f0e885093#file-qlearn-py-L164

    The agent plays a game of catch. Fruits drop from the sky and the agent can choose the actions
    left/stay/right to catch the fruit before it reaches the ground.
"""
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import numpy as np
import random
import math
import os
from time import perf_counter as clock
from cx_Oracle import connect
from datetime import datetime
import PIL.Image as pilimg
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import NotAI
import Create_Model
from random import shuffle

tf.disable_eager_execution()
# tf.enable_eager_execution()

# Parameters

# The probability of choosing a random action (in training). This decays as iterations increase. (0 to 1)
# 무작위 행동을 선택할 확률(훈련 중). 이것은 반복이 증가함에 따라 감소합니다. (0 ~ 1)
epsilon = 1

# The minimum value we want epsilon to reach in training. (0 to 1)
# 훈련에서 엡실론이 도달하기를 원하는 최소값. (0 ~ 1)
epsilonMinimumValue = 0.001

# The number of actions. Since we only have left/stay/right that means 3 actions.
# 작업 수입니다. 왼쪽/머무름/오른쪽만 있으므로 3가지 동작을 의미합니다.
nbActions = 18  # [['', 'z', 'x'], ['', 'left', 'right'], ['', 'down']]의 조합
epoch = 1  # +100  # The number of games we want the system to run for.
hiddenSize = 2000#10  # 20000  # Number of neurons in the hidden layers.
maxMemory = 500  # 메모리의 크기(과거 경험을 저장하는 위치).

# The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.
# 훈련용 미니 배치 크기. 샘플은 미니 배치 크기까지 메모리에서 무작위로 가져옵니다.
batchSize = 50

gridSize_x = 41 # 82  # 24 # 96  # 게임 화면 크기(가로)
gridSize_y = 72 # 41 # 82  # 36 # 144  # 게임 화면 크기(세로)
chanel = 3  # 채널 수
nbStates = gridSize_x * gridSize_y  # We eventually flatten to a 1d tensor to feed the network.
discount = 0.9  # 할인은 네트워크가 보상을 더 빨리 받을 수 있는 상태를 선택하도록 하는 데 사용됩니다(0에서 1).
learningRate = 0.00000001  # 0.2  # Learning Rate for Stochastic Gradient Descent (our optimizer).

# tf2.Variable(tf.random_uniform([len(xData[0]),100],-1, 1, tf2.float64))
# Create the base model.
X = tf.placeholder(tf2.float32, [None, nbStates])
# W1 = tf2.Variable(tf.truncated_normal([nbStates, hiddenSize], stddev=1.0 / math.sqrt(float(nbStates))))
# 정규분포, He 초기값(sqrt(2/n))
W1 = tf2.Variable(tf.random_normal([nbStates, hiddenSize], stddev=math.sqrt(2.0 / float(nbStates))))
b1 = tf2.Variable(tf.random_normal([hiddenSize], stddev=0.01))    
input_layer = tf2.nn.relu(tf2.matmul(X, W1) + b1)

# W2 = tf2.Variable(tf.truncated_normal([hiddenSize, hiddenSize], stddev=1.0 / math.sqrt(float(hiddenSize))))
W2 = tf2.Variable(tf.random_normal([hiddenSize, hiddenSize], stddev=math.sqrt(2.0 / float(hiddenSize))))
b2 = tf2.Variable(tf.random_normal([hiddenSize], stddev=0.01))
hidden_layer = tf2.nn.relu(tf2.matmul(input_layer, W2) + b2)

# W3 = tf2.Variable(tf.truncated_normal([hiddenSize, nbActions], stddev=1.0 / math.sqrt(float(hiddenSize))))
W3 = tf2.Variable(tf.random_normal([hiddenSize, nbActions], stddev=math.sqrt(2.0 / float(hiddenSize))))
b3 = tf2.Variable(tf.random_normal([nbActions], stddev=0.01))
output_layer = tf2.matmul(hidden_layer, W3) + b3

# True labels
Y = tf.placeholder(tf2.float32, [None, nbActions])

# Mean squared error cost function
# 평균 제곱 오차 비용 함수
cost = tf2.reduce_sum(tf2.square(Y - output_layer)) / (2 * batchSize)

# Stochastic Gradient Decent Optimizer
# # 확률적 경사하강법 최적화
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)


# Helper function: Chooses a random value between the two boundaries.
def randf(s, e):
    return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s;


# The environment: Handles interactions and contains the state of the environment
# 환경: 상호작용을 처리하고 환경의 상태를 포함합니다.
class CatchEnvironment():

    def __init__(self, gridSize_x, gridSize_y):
        self.gridSize_x = gridSize_x
        self.gridSize_y = gridSize_y
        self.nbStates = self.gridSize_x * self.gridSize_y  # 판의 넓이(직사각형)
        self.state = np.empty(3, dtype=np.uint8)  # 빈 공간 3개 생성(상태 저장)

    # Returns the state of the environment.
    def observe(self):
        canvas = self.drawState()
        canvas = np.reshape(canvas, (-1, self.nbStates))
        return canvas

    def drawState(self):
        canvas = np.zeros((self.gridSize_x, self.gridSize_y))
        canvas[self.state[0] - 1, self.state[1] - 1] = 1  # Draw the fruit.
        # Draw the basket. The basket takes the adjacent two places to the position of basket.
        canvas[self.gridSize - 1, self.state[2] - 1 - 1] = 1
        canvas[self.gridSize - 1, self.state[2] - 1] = 1
        canvas[self.gridSize - 1, self.state[2] - 1 + 1] = 1        
        return canvas                

    # Resets the environment. Randomly initialise the fruit position (always at the top to begin with) and bucket.
    # 환경을 초기화합니다. 과일 위치(처음에는 항상 맨 위에 있음)와 양동이를 무작위로 초기화합니다.
    def reset(self): 
        initialFruitColumn = random.randrange(1, self.gridSize + 1)
        initialBucketPosition = random.randrange(2, self.gridSize + 1 - 1)
        self.state = np.array([1, initialFruitColumn, initialBucketPosition]) 
        return self.getState()

    def getState(self):
        stateInfo = self.state
        fruit_row = stateInfo[0]
        fruit_col = stateInfo[1]
        basket = stateInfo[2]
        return fruit_row, fruit_col, basket

    # Returns the award that the agent has gained for being in the current environment state.
    def getReward(self):
        fruitRow, fruitColumn, basket = self.getState()
        # If the fruit has reached the bottom.
        # 과일이 바닥에 닿았을 경우.
        if (fruitRow == self.gridSize - 1):
                # Check if the basket caught the fruit.
                # 바구니에 과일이 걸렸는지 확인하세요.
            if (abs(fruitColumn - basket) <= 1):  # 바구니의 중심과 과일의 거리차가 1이하
                return 1  # 걸림
            else:
                return -1  # 놓침
        else:
            return 0  # 내려오는 중

    def isGameOver(self):
        if (self.state[0] == self.gridSize - 1): 
            return True 
        else: 
            return False 

    def updateState(self, action):
        if (action == 1):
            action = -1
        elif (action == 2):
            action = 0
        else:
            action = 1
        fruitRow, fruitColumn, basket = self.getState()
        newBasket = min(max(2, basket + action), self.gridSize - 1)  # The min/max prevents the basket from moving out of the grid.
        fruitRow = fruitRow + 1  # The fruit is falling by 1 every action.
        self.state = np.array([fruitRow, fruitColumn, newBasket])

    # Action can be 1 (move left) or 2 (move right)
    def act(self, action):
        self.updateState(action)
        reward = self.getReward()
        gameOver = self.isGameOver()
        return self.observe(), reward, gameOver, self.getState()  # For purpose of the visual, I also return the state.


class NotTetris2:

    def __init__(self, sess, x1, y1, x2, y2, gridSize_x, gridSize_y, chanel, output,
                 db='choi', db_name='choi', ip='localhost'):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2  # 불러올 이미지 영역
        self.db = db
        self.db_name = db_name
        self.ip = ip
        self.gridSize_x, self.gridSize_y, self.chanel, self.output = gridSize_x, gridSize_y, chanel, output
        
        # self.model = Create_Model.Model(sess, 'model', self.chanel, self.gridSize_y, self.gridSize_x, self.output)
        self.oper = NotAI.Operation()        
        
        # print('n :', self.oper.key_bool_li2number(self.oper.nbActions, [0,1,0,1,0]))
        # print(self.oper.nbActions)
    
    def load_game(self, nc_ng_no):
        try:
            con = connect('%s/%s@%s:1521/xe' % (self.db, self.db_name, self.ip))  # db에 연결
            cur = con.cursor()
            
            sql = """
            select *
            from NotAI_game3
            where ng_no = %d
            """ % nc_ng_no
            # print(sql)
            cur.execute(sql)
            
            self.game_info = None
            for i in cur:
                # print(i)
                self.game_info = {
                    'start_game_time':datetime.strftime(i[1], '%Y%m%d-%H%M%S'),
                    'start_game_clock':i[2],
                    'end_game_time':datetime.strftime(i[3], '%Y%m%d-%H%M%S'),
                    'end_game_clock':i[4]
                    }
            
            print(nc_ng_no, self.game_info)
            
            con.close()
            
            _dir = r'\orbeat\NotAI\data\img\%s_%.4f' % (self.game_info['start_game_time'], self.game_info['start_game_clock'])
            
            con = connect('%s/%s@%s:1521/xe' % (self.db, self.db_name, self.ip))  # db에 연결
            cur = con.cursor()
            
            sql = """
            select *
            from NotAI_Control3
            where nc_ng_no = %d
            order by nc_no
            """ % nc_ng_no
            # print(sql)
            cur.execute(sql)
            
            self.frames = []
            for i in cur:
                # print(i, self.frames)
                key_bool_li = [i[2], i[3], i[4], i[5], i[6]]
                img_path = _dir + '\\%.4f_%s.png' % (i[1], key_bool_li)
                
                cut_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # pilimg.open(img_path)
                # cut_img2= cut_img
                cut_img = cv2.cvtColor(cut_img, cv2.COLOR_BGR2RGB)
                cut_img = cut_img[self.y1:self.y2 + 1, self.x1:self.x2 + 1,:]
                cut_img = NotAI._pooling(2, 2, cut_img, min_sw=True)[:,:, 0]
                cut_img = cut_img.flatten()
                
                # 이전 이미지와 같으면 해당 프레임의 데이터는 학습 데이터에 포함하지 않음
                if len(self.frames) >= 1 and np.all(self.frames[-1]['screenshot']==cut_img):
                    # plt.imshow(cut_img2)
                    # plt.show()
                    continue
                    
                self.frames.append({})
                self.frames[-1]['nc_no'] = i[0]
                self.frames[-1]['current_clock'] = i[1]
                self.frames[-1]['z'] = i[2]
                self.frames[-1]['x'] = i[3]
                self.frames[-1]['left'] = i[4]
                self.frames[-1]['right'] = i[5]
                self.frames[-1]['down'] = i[6]
                self.frames[-1]['score'] = i[7]
                self.frames[-1]['level'] = i[8]
                self.frames[-1]['line'] = i[9]
                self.frames[-1]['next_block'] = i[10]
                self.frames[-1]['key'] = self.oper.key_bool_li2number(self.oper.nbActions, key_bool_li)
                self.frames[-1]['screenshot'] = cut_img
                # print(self.frames[-1])
                # self.frames[-1]['screenshot'] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # pilimg.open(img_path)
                # self.frames[-1]['screenshot'] = cv2.cvtColor(self.frames[-1]['screenshot'], cv2.COLOR_BGR2RGB)
                # # print(self.frames[-1]['screenshot'].shape)
                # self.frames[-1]['screenshot'] = self.frames[-1]['screenshot'][self.y1:self.y2+1, self.x1:self.x2+1, :]
                # # print(self.frames[-1]['screenshot'].shape)
                # self.frames[-1]['screenshot'] = NotAI._pooling(4, 4, self.frames[-1]['screenshot'], min_sw=True)[:,:,0]
                # # print(self.frames[-1]['screenshot'].shape)
                # # self.frames[-1]['screenshot'].show()
                # self.frames[-1]['screenshot'] = self.frames[-1]['screenshot'].flatten()
                # # self.frames[-1]['screenshot'] = np.array(self.frames[-1]['screenshot'])[self.y1:self.y2+1, self.x1:self.x2+1]
                # # print(self.frames[-1]['screenshot'])
                # # plt.imshow(self.frames[-1]['screenshot'])
                # # plt.show()
                
                # self.frames[-1]['screenshot'] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # pilimg.open(img_path)
                # self.frames[-1]['screenshot'] = cv2.cvtColor(self.frames[-1]['screenshot'], cv2.COLOR_BGR2RGB)
                # self.frames[-1]['screenshot'] = self.frames[-1]['screenshot'][self.y1:self.y2 + 1, self.x1:self.x2 + 1,:]
                # self.frames[-1]['screenshot'] = NotAI._pooling(2, 2, self.frames[-1]['screenshot'], min_sw=True)[:,:, 0]
                # self.frames[-1]['screenshot'] = self.frames[-1]['screenshot'].flatten()
                
                # Image.fromarray(self.frames[-1]['screenshot'], 'RGB').show()
                # exit()
            
        except Exception as e:
            print('DB불러오기 실패', e)
            
        try:
            con.close()
        except Exception as e:
            print('DB close 실패', e)
            

# The memory: Handles the internal memory that we add experiences that occur based on agent's actions,
# and creates batches of experiences based on the mini-batch size for training.
# 메모리: 에이전트의 행동에 따라 발생하는 경험을 추가하는 내부 메모리를 처리하고,
# 훈련을 위한 미니 배치 크기를 기반으로 경험 배치를 생성합니다.
class ReplayMemory:

    def __init__(self, gridSize_x, gridSize_y, maxMemory, discount, env):
        self.maxMemory = maxMemory  # 최대 메모리
        self.gridSize_x = gridSize_x  # 게임 화면 가로 길이
        self.gridSize_y = gridSize_y  # 게임 화면 세로 길이
        self.nbStates = self.gridSize_x * self.gridSize_y  # 게임 화면 크기
        self.discount = discount  # 할인율
        canvas = np.zeros((self.gridSize_x, self.gridSize_y))  # 게임 화면을 0으로 채움
        canvas = np.reshape(canvas, (-1, self.nbStates))  # [[0 0 0 0 0 ... 0 0 0 0 0]] 형태로 변환
        
        # 학습을 위한 데이터들을 저장
        self.inputState = np.empty((self.maxMemory, self.nbStates), dtype=np.float32)
        self.actions = np.zeros(self.maxMemory, dtype=np.uint8)
        self.nextState = np.empty((self.maxMemory, self.nbStates), dtype=np.float32)
        self.gameOver = np.empty(self.maxMemory, dtype=np.bool)
        self.rewards = np.empty(self.maxMemory, dtype=np.int8) 
        
        self.count = 0
        self.current = 0
        
        self.env = env

    # Appends the experience to the memory.
    def remember(self, currentState, action, reward, nextState, gameOver):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.inputState[self.current, ...] = currentState
        self.nextState[self.current, ...] = nextState
        self.gameOver[self.current] = gameOver
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.maxMemory

    def getBatch(self, model, batchSize, nbActions, nbStates, sess, X):
        
        # We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        # batch we can (at the beginning of training we will not have enough experience to fill a batch).
        # 전체 배치를 만들기에 충분한 메모리 입력이 있는지 확인합니다. 그렇지 않은 경우 가장 큰
        # 배치 우리는 할 수 있습니다(훈련 초기에는 배치를 채울 만큼 충분한 경험이 없을 것입니다).
        memoryLength = self.count
        chosenBatchSize = min(batchSize, memoryLength)

        inputs = np.zeros((chosenBatchSize, nbStates))
        targets = np.zeros((chosenBatchSize, nbActions))

        # Fill the inputs and targets up.
        # 입력과 대상을 채웁니다.
        # print(chosenBatchSize)
        # print(range(chosenBatchSize))
        for i in range(chosenBatchSize):
            if memoryLength == 1:
                memoryLength = 2
            # Choose a random memory experience to add to the batch.
            # 배치에 추가할 임의의 메모리 경험을 선택합니다.
            randomIndex = random.randrange(1, memoryLength)
            current_inputState = np.reshape(self.inputState[randomIndex], (1, self.nbStates))

            target = sess.run(model, feed_dict={X: current_inputState})
            
            current_nextState = np.reshape(self.nextState[randomIndex], (1, self.nbStates))
            current_outputs = sess.run(model, feed_dict={X: current_nextState})            
            
            # Gives us Q_sa, the max q for the next state.
            # 다음 상태에 대한 최대 q인 Q_sa를 제공합니다.
            nextStateMaxQ = np.amax(current_outputs)
            if (self.gameOver[randomIndex] == True):
                target[0, [self.actions[randomIndex] - 1]] = self.rewards[randomIndex]
            else:
                # reward + discount(gamma) * max_a' Q(s',a')
                # We are setting the Q-value for the action to    r + gamma*max a' Q(s', a'). The rest stay the same
                # to give an error of 0 for those outputs.
                # 액션에 대한 Q-값을 r + gamma*max a' Q(s', a')로 설정합니다. 나머지는 그대로 유지
                # 해당 출력에 대해 0의 오류를 제공합니다.
                target[0, [self.actions[randomIndex] - 1]] = self.rewards[randomIndex] + self.discount * nextStateMaxQ

            # Update the inputs and targets.
            # 입력과 대상을 업데이트합니다.
            inputs[i] = current_inputState
            targets[i] = target

        return inputs, targets
        
        
def main(start, end):
    with tf.Session() as sess: 
        # print("sdf")
        # exit()
        print("Training new model")
    
        # Define Environment
        # 환경 변수(게임 판의 크기, 행동 상태?) 설정
        # env = CatchEnvironment(gridSize_x, gridSize_y)
        # env = NotTetris2(sess, 3, 26, 98, 169, gridSize_x, gridSize_y, chanel, nbActions)
        # env = NotTetris2(sess, 17, 88, 98, 169, gridSize_x, gridSize_y, chanel, nbActions)
        env = NotTetris2(sess, 17, 26, 98, 169, gridSize_x, gridSize_y, chanel, nbActions)
        # t1 = clock()
        # env.load_game(3)
        # for i in env.frames:
            # print(i)
        # print(clock()-t1)
        # exit()
        
        # Define Replay Memory
        memory = ReplayMemory(gridSize_x, gridSize_y, maxMemory, discount, env)
    
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # checkpoint = tf2.train.Checkpoint()
        
        winCount = 0
        # tf.initialize_all_variables().run()
        
        try:
            saver.restore(sess, os.getcwd() + "/model.ckpt")
            print('불러오기 성공')
        except:
            tf.global_variables_initializer().run()
        # ss = sess.run(W1)
        # for i in ss:
            # for j in i:
                # print(j)
        # exit()

        for i in range(start, end + 1):
            # Initialize the environment.
            err = 0
            # env.reset()
            env.load_game(i)
         
            isGameOver = False

            # The initial state of the environment.
            # currentState = env.observe()
            currentState = env.frames[0]['screenshot']
            
            # t3 = clock()
            cnt = 1
            # while (isGameOver != True):
            # print(list(range(len(env.frames))))
            frame_li = list(range(len(env.frames)))
            # shuffle(frame_li)
            for arr, cnt in enumerate(frame_li):
                print('cnt :', cnt)
            # for j in range(1, len(env.frames)-15):
                # t1 = clock()
                # action = -9999  # action initilization
                # Decides if we should choose a random action, or an action from the policy network.
                global epsilon
                # if (randf(0, 1) <= epsilon):
                    # action = random.randrange(1, nbActions + 1)
                # else: 
                    # # Forward the current state through the network.
                    # q = sess.run(output_layer, feed_dict={X: currentState}) 
                    # # Find the max index (the chosen action).
                    # index = q.argmax()
                    # action = index + 1         
                    # print(q, index)
                if True:#arr%100==1:# or __name__=='__main__':
                    q = sess.run(output_layer, feed_dict={X:[currentState]})
                    # q = sess.run(env.model.logits, feed_dict={env.model.X:[currentState]})
                    index = q.argmax()
                    print(q, index)      

                # Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
                # if (epsilon > epsilonMinimumValue):
                    # epsilon = epsilon * 0.999
                
                nextState = env.frames[cnt]['screenshot']
                reward = env.frames[cnt]['score'] - env.frames[cnt - 1]['score']
                gameOver = cnt + 1 >= len(env.frames) - 15
                # stateInfo = #env.act(action)
                
                key = env.frames[cnt]
                if key['x'] or  key['z'] or key['left'] or key['right'] or  key['down']: # 행동을 했는데 점수를 얻지 못한 경우
                    reward -= 1 # 1점 감점
                        
                # if (reward == 1):
                    # winCount = winCount + 1

                memory.remember(currentState, env.frames[cnt]['key'], reward, nextState, gameOver)
                
                # Update the current state and if the game is over.
                # 게임이 종료되면 현재 상태를 업데이트합니다.
                currentState = nextState
                isGameOver = gameOver
                                
                # We get a batch of training data to train the model.
                # 우리는 모델을 훈련하기 위해 훈련 데이터의 배치를 얻습니다.
                t4 = clock()
                inputs, targets = memory.getBatch(output_layer, batchSize, nbActions, nbStates, sess, X)
                # inputs, targets = memory.getBatch(env.model.logits, batchSize, nbActions, nbStates, sess, env.model.X)
                t5 = clock()
                print(i, len(env.frames) - 15, cnt, currentState.shape, inputs.shape, targets.shape, t5 - t4)
                
                # Train the network which returns the error.
                # 오류를 반환하는 네트워크를 훈련시킵니다.
                _, loss = sess.run([optimizer, cost], feed_dict={X: inputs, Y: targets})
                # _, loss = sess.run([env.model.optimizer, env.model.cost], feed_dict={env.model.X: inputs, env.model.Y: targets})    
                err = err + loss
                
                # t2 = clock()
                cnt += 1
                # print(t2-t3, cnt, (t2-t3)/cnt)
                if isGameOver: break

            print("Epoch " + str(i) + ": err = " + str(err) + ": Win count = " + str(winCount) + " Win ratio = " + str(float(winCount) / float(i + 1) * memory.nbStates))
        # Save the variables to disk.
        save_path = saver.save(sess, os.getcwd() + "/model.ckpt")  # 학습 종료후 저장
        # save_path = checkpoint.save('saves/')
        # checkpoint.restore(save_path)
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    main(1, epoch)
    # tf.app.run(1, epoch)  # main 함수로

