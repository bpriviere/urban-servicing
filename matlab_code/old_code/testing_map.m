

% dev map ability
clear all; close all; clc;

my_set_param;



 


% % init_sim;
% 
% % parameters
% 
h = figure(); 
%     axis tight manual
axis square
axis off
set(gcf,'color','w');
mapshow(param.map, 'FaceColor', param.face_color) 
init_sim;
for i = 1:param.Ni
    plot(agents(i).x, agents(i).y,'r*')
end

for i = 1:param.Ns
    map_s = param.my_state_to_map_state(i);
    [x,y] = point_in_poly(param.map(map_s).BoundingBox, param.map(map_s).X, param.map(map_s).Y);
    plot(x,y,'g*');
end


function [X,Y] = coords2pts(lat1, long1, lat2, long2)
    x1 = .18 + (.86-.18) * ((lat1 + 88)/ (-87.5+88));
    y1 = .11 + (.92 - .11) * ((long1 - 41.6)/(42.05 - 41.6));
    x2 = .18 + (.86-.18) * ((lat2 + 88)/ (-87.5+88));
    y2 = .11 + (.92 - .11) * ((long2 - 41.6)/(42.05 - 41.6));
    X = [x1 x2];
    Y = [y1,y2];
end

