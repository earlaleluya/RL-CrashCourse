%% Main.m
% This is the main program for the Inverted Pendulum Simulation.
% This program uses Q-learning algorithm.
% Also, you can select what action strategy to use - exploit or epsilon-greedy.
% Also, it has two modes - training or testing.

%% ------------------------------ SETUP -----------------------------------

clear; close all;

% MODE: 0 if training
%       1 if testing
global MODE; 
MODE = 1;


% ACTION_STRATEGY: 0 if exploit 
%                  1 if epsilon-greedy
global ACTION_STRATEGY;
ACTION_STRATEGY = 1;


% Termination criterion.
MAX_EPISODES = 10000;       
MAX_STEPS    = 2500;

global N_STATES; global ALPHA; global GAMMA; global BETA; global EPSILON;
N_STATES = 162;       % Number of disjoint boxes of state space.
ALPHA   = 0.5;        % Learning rate. 
GAMMA   = 0.85;       % Discount factor for future reward.
BETA    = 0.01;       % Magnitude of noise added to choice.
EPSILON = 0.99;        % Exploitation-Exploration ratio.


% Turning on the double buffering to plot the cart and pole.
h = figure('name','Inverted Pendulum','NumberTitle','off');
set(h,'doublebuffer','on')
   

%% --------------------------- START OF ALGORITHM ------------------------- 

%% 1 Set Q-factor table.
if (MODE==0) % training
    Q = zeros(N_STATES,2);  
else % testing
    load Q;
    EPSILON = 0.0; % Exploit only
end

%% 2 Initial cart and pole.
[x,x_dot,theta,theta_dot,state,action] = reset_cart_pole();

%% Initializing loop variables
steps = 1;
episodes = 1;
reward = 0;


%% Episode-steps loop.
while (steps <= MAX_STEPS && episodes <= MAX_EPISODES)
    %% Display Q-Table
    %Q  %Remove the first % in this line to display Q-table 
    
    %% 3 Get current state.
    state  = get_state(x, x_dot, theta, theta_dot);
    
    %% 4 Determine a action according to equation.
    action = get_action(Q,state);
    
    %% 5.1 Push cart: Apply action to the simulated cart-pole.
    [x,x_dot,theta,theta_dot] = cart_pole(action-1,x,x_dot,theta,theta_dot);
    plot_cart_pole(x,theta,episodes,steps,reward,state,action);
    
    
    %% 5.2 Get current state
    next_state = get_state(x, x_dot, theta, theta_dot);
    
    %% 6 Observe state and decide a reward.
    reward = get_reward(next_state);
    
   %% 7 Update Q based on the formula.
    if (next_state<0) % fail
        Q(state,action) = Q(state,action) + ALPHA*(reward+(GAMMA*0)-Q(state,action));
    else
        Q(state,action) = Q(state,action) + ALPHA*(reward+GAMMA*max(Q(next_state,:))-Q(state,action));
    end
     
    %% Update environmental variables
    if (next_state<0)    % fail
        [x,x_dot,theta,theta_dot,cur_state,cur_action] = reset_cart_pole();
        disp(['EPISODE = ' int2str(episodes), '  STEPS = '  num2str(steps)]);    
        episodes = episodes + 1;  % start new episode
        EPSILON = EPSILON * 0.99; % decaying by 1%
        if (steps~=MAX_STEPS)
            steps = 1;
        end       
    else            % no fail
        steps = steps+1;
    end
      
end


%% Show summary.
if (episodes-1 == MAX_EPISODES)
    disp(['Pole not balanced. Stopping after ' int2str(episodes-1) ' failures ' ]);
else
    disp(['Pole balanced successfully for at least ' int2str(steps-1) ' steps ' ]);
end
