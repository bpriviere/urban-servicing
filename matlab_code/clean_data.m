function agents_data = clean_data(data)
    good_idx = all(data ~= 0,2);
    agents_data = data(good_idx,:);
end