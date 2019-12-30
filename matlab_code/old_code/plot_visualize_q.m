function plot_visualize_q(param, sim_results)


tol = 1;
q_vis = reshape(sim_results.Q(:,1,end), [27,27]) < tol;
figure()
imagesc(q_vis)



end