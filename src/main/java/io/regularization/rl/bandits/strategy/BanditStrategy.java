package io.regularization.rl.bandits.strategy;

import io.regularization.rl.bandits.Bandit;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Created by ptah on 08/02/2017.
 */
public interface BanditStrategy<B extends Bandit> {
    List<B> buildBandits(double weight1, double weight2, double weight3);

    int exploreExploit(List<B> bandits, int iteration);
    default int exploit(List<B> bandits, int iteration) {
        double[] means = bandits.stream().mapToDouble(bandit -> bandit.getMean()).toArray();
        return Nd4j.argMax(Nd4j.create(means)).getInt(0);
    }
}
