
function tts = calcTTS(param,s,a)
    [s_x, s_y] = s_to_loc(param,s);
    tts = norm([a.pickup_x, a.pickup_y] - [s_x, s_y] ) / param.C2;
end