function [sx,sy] = s_to_loc(param,s)

    if param.real_data_on
        map_s = param.my_state_to_map_state(s);
        box = param.map(map_s).BoundingBox;
        sx = mean(box(:,1));
        sy = mean(box(:,2));
    else
        s = s-1;
        i_y = floor( s/ param.Nx);
        i_x = mod( s, param.Nx);
%         if and(i_x == 0, s ~=-1) 
%             i_x = param.Nx;
%         end
        sx = param.scale*(i_x + 0.5);
        sy = param.scale*(i_y + 0.5);
    end

end