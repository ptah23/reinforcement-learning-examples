package io.regularization.rl.bandits.main;

import io.regularization.rl.bandits.ChartBuilder;
import io.regularization.rl.bandits.Experiment;
import io.regularization.rl.bandits.strategy.Thompson;
import io.regularization.rl.bandits.strategy.UCB1;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by ptah on 08/02/2017.
 */
public class ThompsonMain {
    public static void main(String args[]){
        int iterations = 100000;
        double[] c1 = new Experiment(new Thompson()).runExperiment(1.0, 2.0, 3.0, iterations);

        INDArray range = Nd4j.arange(iterations).add(1);
        double[] x = range.data().asDouble();
        XYChart chart = ChartBuilder.buildChart();
        // Series
        chart.addSeries("thompson", x, c1);

        new SwingWrapper<>(chart).displayChart();

    }
}
