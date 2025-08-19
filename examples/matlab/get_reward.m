function [reward]=get_reward(state)

if (state>0) % no fail
    reward = 0.0;
else % fail
    reward = -1.0;
end
