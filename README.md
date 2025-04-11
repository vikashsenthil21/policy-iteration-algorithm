# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.

## POLICY ITERATION ALGORITHM
tep1 :
we are going to do policy evaluation of each state to get the state value function where the initial policy is defined randomly to the mdp.

Step2:
Once we obtain convergence in the policy evaluation then implement policy improvement where we are going to find best optimal policy until the previous and current policy are same.
## POLICY IMPROVEMENT FUNCTION
### Name : VIKASH S
### Register Number : 212222240115
```python
Include the policy improvement function

def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to improve the given policy
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
          new_pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi





```
## POLICY ITERATION FUNCTION
### Name: VIKASH S
### Register Number: 212222240115
```python
Include the policy iteration function

def policy_iteration(P, gamma=1.0, theta=1e-10):
   random_actions=np.random.choice(tuple(P[0].keys()),len(P))
   pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
   while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
   return V, pi




```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
![image](https://github.com/user-attachments/assets/3e4c4952-8c24-46d9-8886-68c673c62617)
![image](https://github.com/user-attachments/assets/11bf433c-bbf9-469e-8e20-7ef55a82b45f)

![image](https://github.com/user-attachments/assets/b87d068b-00be-4051-9a67-9c1817c32df7)


### 2. Policy, Value function and success rate for the Improved Policy
![image](https://github.com/user-attachments/assets/8e1861cb-050d-4af7-b635-373fa5ffd7f1)
![image](https://github.com/user-attachments/assets/21a94fd6-0e26-457a-9dfd-2903c3d8d06d)
![image](https://github.com/user-attachments/assets/85c82aa7-da06-495e-a210-dcfd1ea213f9)


### 3. Policy, Value function and success rate after policy iteration

![image](https://github.com/user-attachments/assets/0beb2599-9bf8-4865-9a75-61232606ad4e)
![image](https://github.com/user-attachments/assets/3c30596b-93c1-440e-b798-18a8e4560049)
![image](https://github.com/user-attachments/assets/1cb669ca-43b4-417f-bb57-af7fb8e891cc)



## RESULT:
Thus, The Python program to find the optimal policy for the given MDP using the policy iteration algorithm is successfully executed.
