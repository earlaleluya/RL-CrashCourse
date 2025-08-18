function [x,x_dot,theta,theta_dot,cur_state,cur_action] = reset_cart_pole()

% Starting state is (0 0 0 0)
x         = 0;       % cart position, meters 
x_dot     = 0;       % cart velocity
theta     = 0;       % pole angle, radians
theta_dot = 0.0;     % pole angular velocity

% Add noise to starting state
global BETA;
x         = x         + random_noise(-BETA, BETA);
x_dot     = x_dot     + random_noise(-BETA, BETA);
theta     = theta     + random_noise(-BETA, BETA);
theta_dot = theta_dot + random_noise(-BETA, BETA);

% 
cur_state = 77;
cur_action = -1;
