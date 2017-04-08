package io.regularization.rl.approximators;

import io.regularization.rl.environment.GridWorldAction;
import io.regularization.rl.environment.GridWorldState;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Model {
    INDArray theta;

    public Model(boolean complex) {
        int size = 4;
        int div = 2;
        if (complex) {
            size = 24;
            div = 5;
        }
        theta = Nd4j.randn(new int[]{size}).divi(div);
    }
    public INDArray s2x(GridWorldState state) {
        float[] x = {state.getI() - 1, state.getJ() - 1.5f, state.getI() * state.getJ() - 3, 1f};
        return Nd4j.create(x).transpose();
    }

    /**
     * more expressive approximation function
     *
     * @param state
     * @param action
     * @return
     */
    public INDArray s2x(GridWorldState state, GridWorldAction action) {
        float[] x = {
                action == GridWorldAction.UP ? state.getI() - 1 : 0,
                action == GridWorldAction.UP ? state.getJ() - 1.5f : 0,
                action == GridWorldAction.UP ? (state.getI() * state.getJ() - 3) / 3 : 0,
                action == GridWorldAction.UP ? (state.getI() * state.getI() - 2) / 2 : 0,
                action == GridWorldAction.UP ? (state.getJ() * state.getJ() - 4.5f) / 4.5f : 0,
                action == GridWorldAction.UP ? 1f : 0,
                action == GridWorldAction.DOWN ? state.getI() - 1 : 0,
                action == GridWorldAction.DOWN ? state.getJ() - 1.5f : 0,
                action == GridWorldAction.DOWN ? (state.getI() * state.getJ() - 3) / 3 : 0,
                action == GridWorldAction.DOWN ? (state.getI() * state.getI() - 2) / 2 : 0,
                action == GridWorldAction.DOWN ? (state.getJ() * state.getJ() - 4.5f) / 4.5f : 0,
                action == GridWorldAction.DOWN ? 1f : 0,
                action == GridWorldAction.LEFT ? state.getI() - 1 : 0,
                action == GridWorldAction.LEFT ? state.getJ() - 1.5f : 0,
                action == GridWorldAction.LEFT ? (state.getI() * state.getJ() - 3) / 3 : 0,
                action == GridWorldAction.LEFT ? (state.getI() * state.getI() - 2) / 2 : 0,
                action == GridWorldAction.LEFT ? (state.getJ() * state.getJ() - 4.5f) / 4.5f : 0,
                action == GridWorldAction.LEFT ? 1f : 0,
                action == GridWorldAction.RIGHT ? state.getI() - 1 : 0,
                action == GridWorldAction.RIGHT ? state.getJ() - 1.5f : 0,
                action == GridWorldAction.RIGHT ? (state.getI() * state.getJ() - 3) / 3 : 0,
                action == GridWorldAction.RIGHT ? (state.getI() * state.getI() - 2) / 2 : 0,
                action == GridWorldAction.RIGHT ? (state.getJ() * state.getJ() - 4.5f) / 4.5f : 0,
                action == GridWorldAction.RIGHT ? 1f : 0
        };
        return Nd4j.create(x).transpose();
    }

    public float predict(GridWorldState state) {
        return theta.mmul(s2x(state)).getFloat(0);
    }

    public float predict(GridWorldState state, GridWorldAction action) {
        return theta.mmul(s2x(state, action)).getFloat(0);
    }

    public INDArray grad(GridWorldState state) {
        return s2x(state);
    }

    public INDArray grad(GridWorldState state, GridWorldAction action) {
        return s2x(state, action);
    }

    public INDArray getTheta() {
        return theta;
    }

    public void setTheta(INDArray theta) {
        this.theta = theta;
    }
}