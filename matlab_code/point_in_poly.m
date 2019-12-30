function [x,y] = point_in_poly(box, poly_x, poly_y)
    found = 0;
    while found == 0
        x = (rand * abs(box(1,1) - box(2,1))) + box(1,1);
        y = (rand * abs(box(1,2) - box(2,2))) + box(1,2);
        if inpolygon(x,y,poly_x, poly_y) == 1
            found = 1;
        end
    end
    
end