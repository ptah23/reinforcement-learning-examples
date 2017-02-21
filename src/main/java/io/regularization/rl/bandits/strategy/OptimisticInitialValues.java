package io.regularization.rl.bandits.strategy;

import io.regularization.rl.bandits.Bandit;
import io.regularization.rl.bandits.Experiment;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by ptah on 04/02/2017.
 */
public class OptimisticInitialValues implements BanditStrategy<Bandit> {
    private double initialValue;


    public OptimisticInitialValues(double initialValue) {
        this.initialValue = initialValue;
    }

    @Override
    public List<Bandit> buildBandits(double weight1, double weight2, double weight3) {
        List<Bandit> bandits = new ArrayList<>();
        bandits.add(new Bandit(weight1, initialValue));
        bandits.add(new Bandit(weight2, initialValue));
        bandits.add(new Bandit(weight3, initialValue));
        return bandits;
    }

    @Override
    public int exploreExploit(List<Bandit> bandits, int iteration) {
       return exploit(bandits, iteration);
    }

}
