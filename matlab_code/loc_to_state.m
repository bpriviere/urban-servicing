function s = loc_to_state( param, x,y)
    

    if param.real_data_on
        
        for i = 1:param.Ns
            map_s = param.my_state_to_map_state(i);
            poly_x = param.map(map_s).X;
            poly_y = param.map(map_s).Y;
            if inpolygon(x,y,poly_x, poly_y) == 1
                s = i;
                break;
            end
        end
    
    else
        s = floor(x/param.scale) + floor(y/param.scale) * param.Nx + 1;
        if x/param.scale == param.Nx
            x = 0.99*x;
            s = floor(x/param.scale) + floor(y/param.scale) * param.Nx + 1;
        end
        if y/param.scale == param.Ny
            y = 0.99*y;
            s = floor(x/param.scale) + floor(y/param.scale) * param.Nx + 1;
        end
    end
        
end