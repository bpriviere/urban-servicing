function R = getR(param)
    R = param.MeasurementNoise*eye(param.Nq);
end