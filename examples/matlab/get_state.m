% get_state:  Given the current state, returns a number from 1 to 162
%   designating the region of the state space encompassing the current state.
%   Returns a value of -1 if a failure state is encountered.

function [state]=get_state(x,x_dot,theta,theta_dot)

one_degree=0.0174532;	% 2pi/360 */
six_degrees=0.1047192;
twelve_degrees=0.2094384;
fifty_degrees=0.87266;


if (x < -2.4 || x > 2.4  || theta < -twelve_degrees || theta > twelve_degrees)          
    state=-1; %/* to signal failure */
else
    
    % Cart Position 
    if (x < -0.8)  		       
        state = 1;
    else
        if (x < 0.8)
            state = 2;
        else		    	               
            state = 3;
        end
    end
    
    % Velocity of cart 
    if (x_dot < -0.5)
        % do nothing
    else
        if (x_dot < 0.5)
            state =state+ 3;
        else 			               
            state =state+ 6;
        end
    end
    
    % Pole angle 
    if (theta < -six_degrees) 	       
       % do nothing 
    else
        if (theta < -one_degree)
            state =state+ 9;
        else
            if (theta < 0)
                state =state+ 18;
            else
                if (theta < one_degree)
                    state=state+ 27;
                else
                    if (theta < six_degrees)
                        state =state+ 36;
                    else	    			       
                        state =state+ 45;
                    end
                end
            end
        end
    end
    
    % Angle velocity of the pole
    if (theta_dot < -fifty_degrees) 	
        % do nothing
    else
        if (theta_dot < fifty_degrees)
            state =state+ 54;
        else                                 
            state =state+ 108;
        end
    end
end