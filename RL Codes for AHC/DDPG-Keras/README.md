
This repository contains the codes for the article "Deep Reinforcement Learning Based Controller for Active Heave Compensation".

Controllers Compared

1. PD Controller
2. Reinforcement Learning based control (DDPG Algorithm)

Cases Analysed

1. No disturbance and no measurement noise
2. Offset Tracking
3. With disturbance but no measurement noise
4. With measurement noise but no disturbance [High noise and Low noise]


Steps to run the code

1. Install all the packages from the requirements.txt file.

2. Make sure that gym package is installed
   To install gym package, type "pip install gym" in the command prompt
   
3. Then install the custom environment which is present inside the folder gym-heave_compensation
    To install the custom environment first go to the desired directory
    Then type cd gym-heave_compensation in the command window to go to the gym-heave_compensation folder
    Then type "pip install -e ." to install that environment
    The gym environment will be installed.
    Reference: https://github.com/openai/gym/blob/master/docs/creating-environments.md
    
4. Once installed open ddpg_winch and run in order to train the agent.

5. Run test_ddpg for testing the RL agent (make sure that the trained weights of the actor and critic network are loaded before testing the RL agent).

