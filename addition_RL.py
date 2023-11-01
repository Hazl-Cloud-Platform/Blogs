import numpy as np 

class Arithmetic():

    def __init__(self):
        self.done = False
        self.mapp = None
        self.observation_space = None
        self.action_space = None
        self.Q = None
        self.score = None
        self.prob = None

    def make_addition(self, a_range, b_range):

        mapp = {}
        state = 0 
        for ai in a_range:
            for bi in b_range:
                mapp[state] = (ai, bi)
                state += 1
        observation_space = list(mapp.keys())
        action_space = list(range(min(a_range)+min(b_range), max(a_range)+max(b_range)+1))
        Q = np.random.rand(len(observation_space), len(action_space))
#        Q = np.zeros((len(observation_space), len(action_space)))
        self.observation_space = observation_space
        self.action_space = action_space
        self.mapp = mapp
        self.Q = Q 

    def map_state2action(self,state):
        (a,b) = self.mapp[state]
        return a+b
    
    def map_ab2state(self,a,b):
        return list(self.mapp.keys())[list(self.mapp.values()).index((a,b))]      

    def exam(self):
        checker = 0
        for state in range(0, self.Q.shape[0]):
            (a,b) = self.mapp[state]
            answer = a + b
            if answer == np.argmax(self.Q[state]):
                checker += 1
                self.prob[state] = max(self.prob)/10
                self.prob = self.prob / sum(self.prob)
        score = checker / self.Q.shape[0]
        self.score = score
        return score

    def quiz(self,a,b):
        state = self.map_ab2state(a,b)
        action = np.argmax(self.Q[state])
        return action

    def reset(self):
        prob = np.ones(len(self.observation_space))/len(self.observation_space)
        state = np.random.choice(
            a = self.observation_space,
            p = prob
            )
        self.prob = prob
        return state

    def next_state(self):
        state = np.random.choice(
            a = self.observation_space,
            p = self.prob
            )
        return state


    def step(self, state, action,counter):
        (a,b) = self.mapp[state]
        if a + b == action:
            reward = 3
        else:
            reward = -1
        state = self.next_state()
        if counter % 5000 == 0:
            if self.exam() >= 0.98:
                done = True
            else:
                done = False
        else:
            done = False
        info = None
        return state, reward, done, info




if __name__ == '__main__':
    
    env = Arithmetic()
    env.make_addition(range(0,31),range(0,31))
    state = env.reset()
    done = False
    counter = 0
    alpha = 0.3
    completeList = []
    
    print ('------------------------------')
    print ('Teach a Machine to Do Addition')
    print ('------------------------------')
    
    print ('')
    quiz = input('Give the machine a quiz? (y/n)')
    
    while quiz in ['y', 'yes', 'Yes']:
        a = input ('a = ')
        b = input ('b = ')
        print ('a + b = ', env.quiz(int(a),int(b)))
        print ('')
        quiz = input('Give another quiz? (y/n)')
        
    
    print ('')
    print ('--------------------')
    print ('The way how RL works')
    print ('--------------------')
    print ('')


    print ('Expert: {} + {} = ?'.format(1,2))
    print ('Machine: {}'.format(5))
    print ('Expert: Wrong, penalty -1 points')
    print ('')
    print ('Expert: {} + {} = ?'.format(2,3))
    print ('Machine: {}'.format(5))
    print ('Expert: Corrent, reward 3 points')    


    print ('')
    input ('Enter to start training...')
    

    
    while done != True:
        
        
        
        action = np.argmax(env.Q[state])
        state2, reward, done, info = env.step(state,action,counter)
        env.Q[state,action] += alpha * (reward +  env.Q[state,action]) #3
        state = state2
        counter += 1
        if counter % 10000 == 0:
            complete = np.round(env.exam()*100,2)
            print ('Complete ', complete , '%')
            completeList.append(complete)
            
    
    print ('')
    quiz = input("Let's test the machine again. (y/n)" )
    
    while quiz in ['y', 'yes', 'Yes']:
        a = input ('a = ')
        b = input ('b = ')
        print ('a + b = ', env.quiz(int(a),int(b)))
        quiz = input('Give it another one? (y/n)')
 