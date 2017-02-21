package io.regularization.rl.bandits;

import io.regularization.rl.bandits.strategy.BanditStrategy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by ptah on 08/02/2017.
 */
public class Experiment {
    BanditStrategy strategy;

    public Experiment(BanditStrategy strategy) {
        this.strategy = strategy;
    }

    public double[] runExperiment(double weight1, double weight2, double weight3,int iterations){
        List<Bandit> bandits = strategy.buildBandits(weight1, weight2, weight3);
        INDArray data = Nd4j.zeros(iterations);
        for(int i = 0; i < iterations; i++) {
            int index = strategy.exploreExploit(bandits, i);
            double x = bandits.get(index).pull();
            data.putScalar(i, x);
        }

        INDArray range = Nd4j.arange(iterations).add(1);
        double[] y = data.cumsumi(0).divi(range).data().asDouble();

        return y;
    }

}
