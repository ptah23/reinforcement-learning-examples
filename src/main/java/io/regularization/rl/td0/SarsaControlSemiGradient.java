package io.regularization.rl.td0;

import com.google.common.collect.Maps;
import io.regularization.rl.approximators.Model;
import io.regularization.rl.dynamicprogramming.IterativePolicyEvaluation;
import io.regularization.rl.environment.GridWorldAction;
import io.regularization.rl.environment.GridWorldEnvironment;
import io.regularization.rl.environment.GridWorldReward;
import io.regularization.rl.environment.GridWorldState;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

/**
 * Created by ptah on 31/03/2017.
 */
public class SarsaControlSemiGradient {
    public static final float ALPHA = 0.1f;
    public static final float GAMMA = 0.9f;
    private static Random random = new Random();

    private static Map<GridWorldAction, GridWorldReward> getQs(Model model, GridWorldState state) {
        Map<GridWorldAction, GridWorldReward> Qs = Maps.newHashMap();
        for (GridWorldAction action : GridWorldAction.values()) {
            float qSa = model.predict(state, action);
            Qs.put(action, new GridWorldReward(qSa));
        }
        return Qs;
    }

    public static void main(String args[]) {
        GridWorldEnvironment grid = GridWorldEnvironment.negativeGrid(-0.1f);
        System.out.println("rewards");
        IterativePolicyEvaluation.printValues(grid.getRewards(), grid);
        Model model = new Model(true);
        float t = 1.0f;
        float t2 = 1.0f;
        List<Float> deltas = new ArrayList<>();
        for (int it = 0; it < 20000; it++) {
            if (it % 100 == 0) {
                t += 10e-3;
                t2 += 0.01;
            }
            if (it % 1000 == 0) {
                System.out.println("it = " + it);
            }
            float alpha = ALPHA / t2;
            GridWorldState state = new GridWorldState(2, 0);
            grid.setCurrentPosition(state);
            Map<GridWorldAction, GridWorldReward> Qs = getQs(model, state);
            GridWorldAction action = max(Qs);
            action = nextAction(action, 0.5f / t);
            float biggestChange = 0f;
            while (!grid.gameOver()) {
                GridWorldReward reward = grid.move(action);
                GridWorldState state2 = grid.getCurrentPosition();
                INDArray oldTheta = model.getTheta();
                if (grid.isTerminal(state2)) {
                    model.setTheta(model.getTheta().add(model.grad(state, action).mul(
                            alpha * (reward.getValue() - model.predict(state, action))
                    )));
                } else {
                    Map<GridWorldAction, GridWorldReward> Qs2 = getQs(model, state2);
                    GridWorldAction action2 = max(Qs2);
                    action2 = nextAction(action2, 0.5f / t);
                    INDArray change = model.grad(state, action).mul(alpha * (reward.getValue() + GAMMA *
                            model.predict(state2, action2) - model.predict(state, action)));
                    model.setTheta(model.getTheta().add(change));
                    state = state2;
                    action = action2;
                }
                biggestChange = Math.max(biggestChange, Transforms.abs(model.getTheta().sub(oldTheta)).sumNumber()
                        .floatValue());
            }
            deltas.add(biggestChange);
        }
        Map<GridWorldState, GridWorldAction> policy = new HashMap<>();
        Map<GridWorldState, GridWorldReward> V = new HashMap<>();
        for (GridWorldState state : grid.getActions().keySet()) {
            Map<GridWorldAction, GridWorldReward> Qs = getQs(model, state);
            GridWorldAction action = max(Qs);
            policy.put(state, action);
            V.put(state, Qs.get(action));
        }

        System.out.println("Values:");
        IterativePolicyEvaluation.printValues(V, grid);
        IterativePolicyEvaluation.printPolicy(policy, grid);


    }

    private static GridWorldAction max(Map<GridWorldAction, GridWorldReward> q) {
        Optional<GridWorldAction> maxValue = q.entrySet().stream()
                .max(
                        Comparator.comparing(
                                entry -> entry.getValue().getValue()
                        )
                ).map(
                        maxResult -> maxResult.getKey()
                );
        return maxValue.get();
    }

    private static GridWorldAction nextAction(GridWorldAction action, float eps) {
        float p = random.nextFloat();
        if (p < (1 - eps)) {
            return action;
        } else {
            return GridWorldAction.values()[random.nextInt(GridWorldAction.values().length)];
        }
    }
}
