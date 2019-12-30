function s = loc_to_state_pp( param, x,y)
    
    s = -1;
    for i = 1:param.Ns_map
        poly_x = param.map(i).X;
        poly_y = param.map(i).Y;
        if inpolygon(x,y,poly_x, poly_y) == 1
            s = i;
            break;
        end
    end
    
end