#-*- coding: utf-8 -*-
""" Kapathy의 DL 코드를 실행해보고 코드 이해가 어려운 중요 부분에 설명을 추가 하였음. 
    2016/07 by funmv """
""" 학습 초기에는 대부분 agent가 패함. 대략 맥북에서 이틀 정도 학습하니 agent가 이기는 경우가 나타나기 시작하였음 """
""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume: # 이 값이 셋 되었으므로 이전에 저장된 값으로 시작 
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

"""
r값은 학습 초기에는 +1(agent가 이기는 것 표현)이 거의 없고, 대부분은 0(승패없이 게임 중)이며, 중간중간에 -1(컴 승)이 있는 값이다.
이런 값들이 r내에 1500~2000개 넘어온다. 학습이 진행되어 갈 수록 +1이 빈번하게 나타나기 시작한다
루프를 보면 중간에 -1을 만나면 running_add=0이 되므로 다시 계산이 된다는 리셋 의미로, 분석 시는 이 라인은 제외하고 분석하면 이해 쉬움
수식으로 쓰보면, ra = rag+r_t -> (rag+rt)*g+r{t+1} 형태의 재귀 폼이다. 두 단계만 전개해보면
((rag+rt)g+r_t+1)g+r_t+2 = (rag^2+rtg+r_t+1)g+r_t+2 = rag^3+rtg^2+r_t+1*g+r_t+2이고 초기 action에서 멀어지면 약하게 가중되는 수식이다
"""  

def policy_forward(x):
  h = np.dot(model['W1'], x)  # W1=[200x6400], x=[6400]
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

"""
eph=Mx200(히든층인 h가 에피소드수만큼 누적된 값), epdlogp=Mx1, dW2=200x1이고 dW2를 구하기 위해 h.W2=Out를 이용. h.dW2 = dOut, dW2= h'.dOut
model['W2']=200, dh=Mx200: 이번에는 dh를 구하기 위한 중간과정으로 다시 상기식 사용. h.W2=Out, dh.W2=dOut, 여기서 dh를 알면 이를 통해
dW1 계산 가능. h=W1.x, dh=dW1.x, dW1=dh'.x
"""


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation) #observation은 이미지 데이터이다(처리된 후 80x80)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
  """ 이 값이 무슨 특별한 의미가 있는 것이 아니라 foward net 출력 확률값이 높은 것을 선호한다는 의미이다. """

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
  """ regression분류이고, 원래 (y-aprob)^2가 Loss이다. 이를 미분한 것이 여기 수식이다. (reg Loss가 제곱인 이유는 미분하면 형태가 단순해지기 떄문) """


  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward
  """ 게임이 끝났을 때 처음에는 주로 지므로 reward가 -1이 대부분. 학습이 진행될수록 +1이 많아진다. 한 에피소드 내에서는 게임 진행중을 의미하는 0이 대부분이고, 중간중간에 -1, +1이 나옴. 맨 끝에는 +1이나 -1
      이런 값들이 1500~2000개가 나온 후 done이 true가 된다. """

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs) # Mx6400, M: 판(에피소드)이 끝난 게임에서 입력(영상)의 갯수 
    eph = np.vstack(hs) # Mx200, 1판은 끝날 때까지 1000~2000개의 이미지(80x80 pxls) 가짐 
    epdlogp = np.vstack(dlogps) # Mx1
    epr = np.vstack(drs) # Mx1
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    """ policy gradient의 마법이 여기서 일어난다고 했는데, 이 식은 기대치(blog에서 f(x))와 policy 확률의 곱이다. """

    grad = policy_backward(eph, epdlogp)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.iteritems():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')

