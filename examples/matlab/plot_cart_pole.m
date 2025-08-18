function plot_cart_pole(x,theta,episode,steps,reward,state,action)


l=2; %pole's Length for ploting it can be different from the actual length


pxg = [x+1 x-1 x-1 x+1 x+1];
pyg = [0.25 0.25 1.25 1.25 0.25];

pxp=[x x+l*sin(theta)];
pyp=[1.25 1.25+l*cos(theta)];

[pxw1,pyw1] = plotcircle(x-0.5,0.125,0,0.125);
[pxw2,pyw2] = plotcircle(x+0.5,0.125,0,0.125);

plot(pxg,pyg,'k-',pxw1,pyw1,'k',pxw2,pyw2,'k',pxp,pyp,'r')
axis([-6 6 0 6])



grid;
global MODE;
if (MODE==0)
    title('Inverted Pendulum: Training');
else
    title('Inverted Pendulum: Testing');
end

text(0.8*l, 2.9*l, 'Algorithm: Q');
    

global ACTION_STRATEGY;
if (ACTION_STRATEGY==0)
    text(0.8*l, 2.7*l, 'Action strategy: exploit');
else
    text(0.8*l, 2.7*l, 'Action strategy: eGreedy');
end    


out1 = ['Episode ', num2str(episode), ':  Step ', num2str(steps)];
text(-2.9*l, 2.9*l, out1);
out1 = ['State = ', num2str(state)];
text(-2.9*l, 2.7*l, out1);

if (action==1)
    out1 = 'Action = left';
else
    out1 = 'Action = right';
end
text(-2.9*l, 2.5*l, out1);

out1 = ['Reward = ', num2str(reward)];
text(-2.9*l, 2.3*l, out1);

global EPSILON;
out1 = ['Epsilon = ', num2str(EPSILON)];
text(-2.9*l, 2.1*l, out1);

drawnow;