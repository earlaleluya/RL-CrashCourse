function [action]=get_action(Q,state)

global EPSILON;
global ACTION_STRATEGY;
global BETA;

% Exploit: get argument maximum.
if (Q(state,1)+random_noise(-BETA, BETA) <= Q(state,2)) 
    action = 2;  % right
else
    action = 1;  % left
end

% Epsilon-Greedy.
if (ACTION_STRATEGY==1 && rand(1)<EPSILON)   
        action = randi([1 2]);  % Explore through random action.
end