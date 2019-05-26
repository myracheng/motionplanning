for i = 2900:3000
    x = data{i}.Observation;
    plot(x(:,1), x(:,2))
    xlim([0,50])
    ylim([0,50])
end