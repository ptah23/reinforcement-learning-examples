package io.regularization.rl.bandits.strategy;

import io.regularization.rl.bandits.Bandit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by ptah on 08/02/2017.
 */
public class UCB1 implements BanditStrategy<Bandit>{

    @Override
    public List<Bandit> buildBandits(double weight1, double weight2, double weight3) {
        List<Bandit> bandits = new ArrayList<>();
        bandits.add(new Bandit(weight1, 0));
        bandits.add(new Bandit(weight2, 0));
        bandits.add(new Bandit(weight3, 0));
        return bandits;
    }

    @Override
    public int exploreExploit(List<Bandit> bandits, int iteration) {
        return exploit(bandits, iteration);
    }

    @Override
    public int exploit(List<Bandit> bandits, int iteration) {
        double[] means = bandits.stream()
                .mapToDouble(bandit -> ucb(bandit.getMean(), iteration + 1, bandit.getN()))
                .toArray();
        return Nd4j.argMax(Nd4j.create(means)).getInt(0);
    }

    private double ucb(double mean, int N, int Nj) {
        return mean + (Math.sqrt(2 * Math.log(N) / (Nj + 10e-3)));
    }

}
