package io.regularization.rl.bandits.strategy;

import io.regularization.rl.bandits.Bandit;
import io.regularization.rl.bandits.BayesianBandit;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by ptah on 11/02/2017.
 */
public class Thompson implements BanditStrategy<BayesianBandit> {

    @Override
    public List<BayesianBandit> buildBandits(double weight1, double weight2, double weight3) {
        List<BayesianBandit> list = new ArrayList<>();
        list.add(new BayesianBandit(weight1));
        list.add(new BayesianBandit(weight2));
        list.add(new BayesianBandit(weight3));
        return list;
    }

    @Override
    public int exploreExploit(List<BayesianBandit> bandits, int iteration) {
        return exploit(bandits, iteration);
    }

    @Override
    public int exploit(List<BayesianBandit> bandits, int iteration) {
        double[] samples = bandits.stream().mapToDouble(bandit -> bandit.sample()).toArray();
        return Nd4j.argMax(Nd4j.create(samples)).getInt(0);
    }
}
