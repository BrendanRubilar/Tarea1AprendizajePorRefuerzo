import numpy as np
import gym
import gym_gridworld
from time import sleep

import os
import pandas as pd
import matplotlib.pyplot as plt

env = gym.make("GridWorld-v0")
env.verbose = True
_ =env.reset()

EPISODES = 10000 
MAX_STEPS = 100 

#Hiperparámetros optimos encontrados 
LEARNING_RATE = 0.05
GAMMA = 1

#Para el problema 4 :)
LAMBDA = 0.5  #Corresponde a la traza de elegibilidad

epsilon = 1

print('Observation space\n')
print(env.observation_space)

print('Action space\n')
print(env.action_space)

def qlearning(env, epsilon):
    STATES =  env.n_states #Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions #Cantidad de acciones posibles.

    Q = np.zeros((STATES, ACTIONS)) #Inicializa la Q table con 0s.
    rewards = []
    for episode in range(EPISODES):
        rewards_epi=0
        state = env.reset() #Reinicia el ambiente
        for actual_step in range(MAX_STEPS):

            if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
                action = env.action_space.sample() 
            else:
                action = np.argmax(Q[state, :]) #De lo contrario, escogerá el estado con el mayor valor.

            next_state, reward, done, _ = env.step(action) #Ejecuta la acción en el ambiente y guarda los nuevos parámetros (estado siguiente, recompensa, ¿terminó?).
            rewards_epi=rewards_epi+reward

            Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action]) #Calcula la nueva Q table.

            state = next_state

            if (MAX_STEPS-2)<actual_step:
            	print (f"Episode {episode} rewards: {rewards_epi}")
            	print(f"Value of epsilon: {epsilon}") 
            	if epsilon > 0.1: epsilon -= 0.0001

            if done:
                print (f"Episode {episode} rewards: {rewards_epi}") 
                #rewards.append((episode, rewards_epi)) #Guarda las recompensas en una lista + el episodio
                print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1: epsilon -= 0.0001
                break
        
        rewards.append((episode, rewards_epi)) #Guarda las recompensas en una lista + el episodio
        
    #Guardar episodios y recompensas en CSV y graficarlos
    csv_path = os.path.join(os.getcwd(), "QLearning.csv")
    df = pd.DataFrame(rewards, columns=['episode', 'reward'])
    df.to_csv(csv_path, index=False)
    print(f"Saved training rewards to {csv_path}")
    plot_rewards_csv(csv_path)

    print(Q)
    return Q

def sarsa(env, epsilon):
    rewards = []
    STATES =  env.n_states #Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions #Cantidad de acciones posibles
    
    Q = np.zeros((STATES, ACTIONS))
    for episode in range(EPISODES):
        rewards_epi=0
        state = env.reset() #Reinicia el ambiente
        
        if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
            action = env.action_space.sample() 
        else:
            action = np.argmax(Q[state, :]) #De lo contrario, escogerá el estado con el mayor valor.

        for actual_step in range(MAX_STEPS):

            next_state, reward, done, _ = env.step(action)
            
            if np.random.uniform(0, 1) < epsilon: #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
                action2 = env.action_space.sample() 
            else:
                action2 = np.argmax(Q[next_state, :]) #De lo contrario, escogerá el estado con el mayor valor.

            Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * Q[next_state, action2] - Q[state, action]) #Calcula la nueva Q table.
            rewards_epi=rewards_epi+reward
            state = next_state
            action = action2

            if (MAX_STEPS-2)<actual_step:
            	print (f"Episode {episode} rewards: {rewards_epi}")
            	print(f"Value of epsilon: {epsilon}") 
            	if epsilon > 0.1: epsilon -= 0.0001

            if done:
                print (f"Episode {episode} rewards: {rewards_epi}") 
                #rewards.append(rewards_epi) #Guarda las recompensas en una lista
                print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1: epsilon -= 0.0001
                break  
            
        rewards.append((episode, rewards_epi)) #Guarda las recompensas en una lista + el episodio
        
    #Guardar episodios y recompensas en CSV y graficarlos
    csv_path = os.path.join(os.getcwd(), "Sarsa.csv")
    df = pd.DataFrame(rewards, columns=['episode', 'reward'])
    df.to_csv(csv_path, index=False)
    print(f"Saved training rewards to {csv_path}")
    plot_rewards_csv(csv_path)

    print(Q)
    return Q

#-----------------------Problema 3: Double Q-Learning---------------------------
def double_qlearning(env, epsilon):
    STATES = env.n_states
    ACTIONS = env.n_actions

    #Cambio 1: Inicialización de dos Tablas Q
    Q1 = np.zeros((STATES, ACTIONS))
    Q2 = np.zeros((STATES, ACTIONS))
    
    rewards = []
    for episode in range(EPISODES):
        rewards_epi = 0
        state = env.reset()
        
        for actual_step in range(MAX_STEPS):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                #Cambio 2: Selección de Acciones
                action = np.argmax(Q1[state, :] + Q2[state, :])

            next_state, reward, done, _ = env.step(action)
            rewards_epi += reward

            #Cambio 3: Separar la Selección de la evaluación
            
            #Se elige al azar cuál tabla actualizar, usando la otra para evaluar
            if np.random.uniform(0, 1) < 0.5:
                #Actualizar Q1 usando el valor de Q2
                best_next_action = np.argmax(Q1[next_state, :]) #Selecciona la acción con Q1
                target_value = Q2[next_state, best_next_action] #Evalúa la acción con Q2
                
                # Fórmula de actualización para Q1
                Q1[state, action] = Q1[state, action] + LEARNING_RATE * \
                    (reward + GAMMA * target_value - Q1[state, action])
            else:
                #Actualizar Q2 usando el valor de Q1
                best_next_action = np.argmax(Q2[next_state, :]) #Selecciona la acción con Q2
                target_value = Q1[next_state, best_next_action] #Evalúa la acción con Q1

                #Fórmula de actualización para Q2
                Q2[state, action] = Q2[state, action] + LEARNING_RATE * \
                    (reward + GAMMA * target_value - Q2[state, action])
            
            state = next_state

            if (MAX_STEPS - 2) < actual_step:
                print(f"Episode {episode} rewards: {rewards_epi}")
                print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1:
                    epsilon -= 0.0001

            if done:
                print(f"Episode {episode} rewards: {rewards_epi}")
                print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1:
                    epsilon -= 0.0001
                break
        
        rewards.append((episode, rewards_epi))

    # Guardar episodios y recompensas en CSV (con un nuevo nombre)
    csv_path = os.path.join(os.getcwd(), "DoubleQlearning.csv")
    df = pd.DataFrame(rewards, columns=['episode', 'reward'])
    df.to_csv(csv_path, index=False)
    print(f"Saved training rewards to {csv_path}")
    
    plot_rewards_csv(csv_path) 

    plot_two_rewards_csv(csv_path, "Qlearning.csv")
    
    print("Q1 Table:")
    print(Q1)
    print("\nQ2 Table:")
    print(Q2)
    
    #En este caso se retornan ambas tablas Q
    return Q1, Q2

#-----------------------Problema 4:  Sarsa(Lambda) y Q(Lambda)---------------------------

def sarsa_lambda(env, epsilon, lam=LAMBDA):
    STATES = env.n_states
    ACTIONS = env.n_actions

    Q = np.zeros((STATES, ACTIONS))
    for_episode_rewards = []

    for episode in range(EPISODES):
        e = np.zeros_like(Q)  #Representa las trazas de elegibilidad
        rewards_epi = 0
        state = env.reset()

        #Primero se selecciona acción inicial usando epsilon-greedy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        for actual_step in range(MAX_STEPS):
            next_state, reward, done, _ = env.step(action)
            rewards_epi += reward

            #Luego se elege la siguiente acción
            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state, :])

            #Se calcula el error de prediccicón
            delta = reward + GAMMA * Q[next_state, next_action] - Q[state, action]

            #Y se actualizan las trazas y Q
            e[state, action] += 1.0
            Q += LEARNING_RATE * delta * e
            e *= GAMMA * lam

            state, action = next_state, next_action

            if (MAX_STEPS - 2) < actual_step:
                if epsilon > 0.1:
                    epsilon -= 0.0001

            if done:
                if epsilon > 0.1:
                    epsilon -= 0.0001
                break

        for_episode_rewards.append((episode, rewards_epi))

    #Guardar episodios y recompensas en CSV y graficarlos
    csv_path = os.path.join(os.getcwd(), "training_rewards_sarsa_lambda.csv")
    df = pd.DataFrame(for_episode_rewards, columns=['episode', 'reward'])
    df.to_csv(csv_path, index=False)
    print(f"Saved training rewards to {csv_path}")
    plot_rewards_csv(csv_path)

    return Q


def q_lambda(env, epsilon, lam=LAMBDA):
    STATES = env.n_states
    ACTIONS = env.n_actions

    Q = np.zeros((STATES, ACTIONS))
    episode_rewards = []

    for episode in range(EPISODES):
        e = np.zeros_like(Q)
        rewards_epi = 0
        state = env.reset()

        for actual_step in range(MAX_STEPS):
            #Seleccionar acción actual usando epsilon-greedy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, done, _ = env.step(action)
            rewards_epi += reward

            #Aquó se hace el cálculo del target usando max (Formula Q-learning)
            best_next = np.argmax(Q[next_state, :])
            delta = reward + GAMMA * Q[next_state, best_next] - Q[state, action]

            #Se incrementa traza y actualizan todas las Q
            e[state, action] += 1.0
            Q += LEARNING_RATE * delta * e

            #Luego se elige la acción que se tomará en el siguiente paso
            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state, :])

            #Del pseudocódigo de Watkins se toma lo siguiente: si la acción elegida no es greedy respecto a Q,
            #resetear trazas, en caso contrario, decaer trazas.
            if next_action != best_next:
                e[:] = 0.0
            else:
                e *= GAMMA * lam

            state = next_state

            if (MAX_STEPS - 2) < actual_step:
                if epsilon > 0.1:
                    epsilon -= 0.0001

            if done:
                if epsilon > 0.1:
                    epsilon -= 0.0001
                break

        episode_rewards.append((episode, rewards_epi))

    #Guardar episodios y recompensas en CSV y graficarlos
    csv_path = os.path.join(os.getcwd(), "QLambda.csv")
    df = pd.DataFrame(episode_rewards, columns=['episode', 'reward'])
    df.to_csv(csv_path, index=False)
    print(f"Saved training rewards to {csv_path}")
    plot_rewards_csv(csv_path)

    return Q

#----------------------AUXILIARES--------------------------------------------------------------------------------
#Funciones utilizadas para generar los graficos de recompensas
def plot_rewards_csv(csv_path, window=50, smoothing='moving', show_raw=True, show_smoothed=True):
    df = pd.read_csv(csv_path)
    df = df.sort_values('episode')
    episodes = df['episode']
    rewards = df['reward']

    if smoothing == 'moving':
        smooth = rewards.rolling(window=window, min_periods=1).mean()
    elif smoothing == 'median':
        smooth = rewards.rolling(window=window, min_periods=1).median()
    elif smoothing == 'ewm':
        smooth = rewards.ewm(span=window, adjust=False).mean()
    else:
        smooth = rewards

    plt.figure(figsize=(10,4))
    if show_raw:
        plt.plot(episodes, rewards, color='C0', alpha=0.25, linewidth=1, label='raw')
    if show_smoothed:
        label = 'smoothed' if smoothing else 'no smoothing'
        plt.plot(episodes, smooth, color='C1', linewidth=1.5, label=f'{smoothing or "raw"} (w={window})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title("Recompensas por Episodio")
    plt.grid(True)
    plt.legend()

    png_path = csv_path.replace('.csv', f'_{smoothing}_w{window}.png') if smoothing else csv_path.replace('.csv', '.png')
    plt.savefig(png_path, bbox_inches='tight')
    print(f"Saved plot to {png_path}")
    plt.show()
    
def plot_two_rewards_csv(csv_path1, csv_path2, window=50, label1=None, label2=None):

    df1 = pd.read_csv(csv_path1).sort_values('episode')
    df2 = pd.read_csv(csv_path2).sort_values('episode')

    ep1 = df1['episode']
    ep2 = df2['episode']
    r1 = df1['reward']
    r2 = df2['reward']

    s1 = r1.rolling(window=window, min_periods=1).mean()
    s2 = r2.rolling(window=window, min_periods=1).mean()

    lbl1 = label1 or os.path.basename(csv_path1).replace('.csv','')
    lbl2 = label2 or os.path.basename(csv_path2).replace('.csv','')

    plt.figure(figsize=(10,4))
    plt.plot(ep1, s1, color='red', linewidth=1.8, label=f'{lbl1} (MA w={window})')
    plt.plot(ep2, s2, color='black', linewidth=1.8, label=f'{lbl2} (MA w={window})')
    plt.xlabel('Episode')
    plt.ylabel('Reward (moving average)')
    plt.title(f'Medias móviles: {lbl1} vs {lbl2}')
    plt.grid(True)
    plt.legend()

    png_name = f'comparison_{lbl1}_vs_{lbl2}_w{window}.png'
    png_path = os.path.join(os.getcwd(), png_name)
    plt.savefig(png_path, bbox_inches='tight')
    print(f"Saved comparison plot to {png_path}")
    plt.show()
    
#--------------------MAIN------------------------------------------------------------------------

#Función para correr juegos siguiendo una determinada política
def playgames(env, Q, num_games, render = True):
    wins = 0
    env.reset()
    #pause=input()
    env.render()

    for i_episode in range(num_games):
        rewards_epi=0
        observation = env.reset()
        t = 0
        while True:
            action = np.argmax(Q[observation, :]) #La acción a realizar esta dada por la política
            observation, reward, done, info = env.step(action)
            rewards_epi=rewards_epi+reward
            if render: env.render()
            #pause=input() 
            sleep(0.1) #Comentar para realizar todos los movimientos sin pausas.
            if done:
                if reward >= 0:
                    wins += 1
                print(f"Episode {i_episode} finished after {t+1} timesteps with reward {rewards_epi}")
                break
            t += 1
    #pause=input()
    sleep(0.1) #Comentar para realizar todos los movimientos sin pausas. 
    env.close()
    print("Victorias: ", wins)


#Descomentar el algoritmo a utilizar y la función para ejecutar el juego :)

#Q = sarsa(env, epsilon)
#Q = qlearning(env, epsilon)

#Q,Q2 = double_qlearning(env, epsilon)
#Q= q_lambda(env, epsilon, LAMBDA)
#Q = sarsa_lambda(env, epsilon, LAMBDA)

#playgames(env, Q, 100, True)
env.close()

#_ =env.step(env.action_space.sample())
