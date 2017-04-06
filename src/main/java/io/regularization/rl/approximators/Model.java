package io.regularization.rl.approximators;

import io.regularization.rl.environment.GridWorldState;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Model {
    INDArray theta = Nd4j.randn(new int[]{4}).divi(2);

    public INDArray s2x(GridWorldState state) {
        float[] x = {state.getI() - 1, state.getJ() - 1.5f, state.getI() * state.getJ() - 3, 1f};
        return Nd4j.create(x).transpose();
    }

    public float predict(GridWorldState state) {
        return theta.mmul(s2x(state)).getFloat(0);
    }

    public INDArray grad(GridWorldState state) {
        return s2x(state);
    }

    public INDArray getTheta() {
        return theta;
    }

    public void setTheta(INDArray theta) {
        this.theta = theta;
    }
}