
# get_maxQ（）的多种实现对比
def get_maxQ(self, state):
    """ The get_max_Q function is called when the agent is asked to find the
        maximum Q-value of all actions based on the 'state' the smartcab is in. """
    ###########
    ## TO DO ##
    ###########
    # Calculate the maximum Q-value of all actions for a given state

    # 实现之一
    maxQ = []
    Q = max(self.Q[state], key=self.Q[state].get)
    for a in self.Q[state]:
        if self.Q[state][a] == self.Q[state][Q]:
            maxQ.append(a)
    return random.choice(maxQ)

    # 实现之二
    max_Q_value = []
    for k, v in self.Q[state].items():
        if v == max(self.Q[state].values()):
            max_Q_value.append(k)
    return random.choice(max_Q_value)