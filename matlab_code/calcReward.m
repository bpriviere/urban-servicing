function R_sa = calcReward(param, s, a)
    tts = calcTTS(param,s,a);
    R_sa =  a.revenue - param.C1*(tts + a.ttc); 
end