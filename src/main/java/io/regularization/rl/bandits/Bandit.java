package io.regularization.rl.bandits;

import java.util.Random;

/**
 * Created by ptah on 04/02/2017.
 */
public class Bandit {
    private double weight;
    private double m = 0.0f;
    private int N = 0;
    private double mean;
    protected Random random = new Random();

    public Bandit(double weight, double initialValue){
        this.weight = weight;
        this.mean = initialValue;
    }

    public double pull() {
        double returnValue =  random.nextGaussian() + getWeight();
        N++;
        mean =(1 - (1.0 / N)) * mean + (1.0 / N) * returnValue;
        return returnValue;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getM() {
        return m;
    }

    public void setM(double m) {
        this.m = m;
    }

    public int getN() {
        return N;
    }

    public void setN(int n) {
        N = n;
    }

    public double getMean() {
        return mean;
    }

    public void setMean(double mean) {
        this.mean = mean;
    }
}
