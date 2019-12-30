function Q_norm = normalizeQ(param, agents, i)

    Q_max = max(agents(i).Q);
    Q_min = min(agents(i).Q);
    Q_norm = (agents(i).Q - Q_min) / (Q_max - Q_min);
    
end