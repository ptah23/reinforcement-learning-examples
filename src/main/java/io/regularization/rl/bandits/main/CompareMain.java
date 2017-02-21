package io.regularization.rl.bandits.main;

import io.regularization.rl.bandits.ChartBuilder;
import io.regularization.rl.bandits.Experiment;
import io.regularization.rl.bandits.strategy.EpsilonGreedy;
import io.regularization.rl.bandits.strategy.OptimisticInitialValues;
import io.regularization.rl.bandits.strategy.Thompson;
import io.regularization.rl.bandits.strategy.UCB1;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by ptah on 11/02/2017.
 */
public class CompareMain {
    public static void main(String[] args){
        int iterations = 100000;

        double[] eps = new Experiment(new EpsilonGreedy()).runExperiment(1.0, 2.0, 3.0, iterations);
        double[] oiv = new Experiment(new OptimisticInitialValues(10)).runExperiment(1.0, 2.0, 3.0, iterations);
        double[] thompson = new Experiment(new Thompson()).runExperiment(1.0, 2.0, 3.0, iterations);
        double[] ucb1 = new Experiment(new UCB1()).runExperiment(1.0, 2.0, 3.0, iterations);

        INDArray range = Nd4j.arange(iterations).add(1);
        double[] x = range.data().asDouble();
        XYChart chart = ChartBuilder.buildChart();
        // Series
        chart.addSeries("decaying epsilon greedy", x, eps);
        chart.addSeries("optimistic initial values", x, oiv);
        chart.addSeries("bayesian", x, thompson);
        chart.addSeries("ucb1", x, ucb1);

        new SwingWrapper<>(chart).displayChart();

    }
}
