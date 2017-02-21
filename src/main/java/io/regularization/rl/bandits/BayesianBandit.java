package io.regularization.rl.bandits;

/**
 * Created by ptah on 11/02/2017.
 */
public class BayesianBandit extends Bandit {
    double lambda0 = 1;
    double m0 = 0;
    double sumX = 0;
    double tau = 1;
    public BayesianBandit(double weight) {
        super(weight, 0);
    }


    public double sample() {
        return random.nextGaussian() / Math.sqrt(lambda0) + m0;
    }

    @Override
    public double pull() {
        double returnValue =  random.nextGaussian() + getWeight();
        lambda0 += 1;
        sumX += returnValue;
        m0 = tau * sumX / lambda0;
        return returnValue;
    }
}
