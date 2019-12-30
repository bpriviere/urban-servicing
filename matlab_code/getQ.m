function Q = getQ(param)
    Q = param.ProcessNoise*eye(param.Nq);
end