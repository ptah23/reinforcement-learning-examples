package io.regularization.rl.bandits.strategy;

import io.regularization.rl.bandits.Bandit;
import io.regularization.rl.bandits.Experiment;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by ptah on 04/02/2017.
 */
public class EpsilonGreedy implements BanditStrategy<Bandit> {

    double eps = 0;

    public EpsilonGreedy(double eps) {
        this.eps = eps;
    }

    public EpsilonGreedy() {

    }


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
        double p = Nd4j.getRandom().nextFloat();
        int index;
        if(p < (eps==0?1.0/(iteration+1):eps)) {
            index = Nd4j.getRandom().nextInt(3);
        } else {
            index = exploit(bandits, iteration);
        }
        return index;
    }

}
