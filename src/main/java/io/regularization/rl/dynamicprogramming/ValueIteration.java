package io.regularization.rl.dynamicprogramming;

import com.google.common.collect.Maps;
import io.regularization.rl.environment.GridWorldAction;
import io.regularization.rl.environment.GridWorldEnvironment;
import io.regularization.rl.environment.GridWorldReward;
import io.regularization.rl.environment.GridWorldState;

import java.util.Map;
import java.util.Random;

/**
 * Created by ptah on 26/02/2017.
 */
public class ValueIteration {
    private static float GAMMA = 0.9f, SMALL_ENOUGH = 10e-4f;

    public static void main(String args[]) {
        Random random = new Random();
        GridWorldEnvironment grid = GridWorldEnvironment.negativeGrid();
        System.out.println("rewards:");
        IterativePolicyEvaluation.printValues(grid.getRewards(), grid);
        //randomly choose action and update as we learn
        Map<GridWorldState, GridWorldAction> policy = Maps.newHashMap();
        for (GridWorldState state : grid.getActions().keySet()) {
            policy.put(state, GridWorldAction.values()[random.nextInt(GridWorldAction.values().length)]);
        }
        System.out.println("initial policy");
        IterativePolicyEvaluation.printPolicy(policy, grid);
        Map<GridWorldState, GridWorldReward> V = IterativePolicyEvaluation.initialiseV(grid);
        while (true) {
            float biggestChange = 0;
            for (GridWorldState state : grid.allStates()) {
                GridWorldReward oldV = V.get(state);
                if (policy.containsKey(state)) {
                    float newV = Float.NEGATIVE_INFINITY;
                    GridWorldAction bestAction = null;
                    for (GridWorldAction action : GridWorldAction.values()) {
                        grid.setCurrentPosition(state);
                        GridWorldReward reward = grid.move(action);
                        float v = reward.getValue() + GAMMA * V.get(grid.getCurrentPosition()).getValue();
                        if (v > newV) {
                            newV = v;
                            bestAction = action;
                        }
                    }
                    V.put(state, new GridWorldReward(newV));
                    policy.put(state, bestAction);
                    biggestChange = Math.max(biggestChange, Math.abs(oldV.getValue() - newV));

                }
            }
            if (biggestChange < SMALL_ENOUGH) {
                break;
            }
        }
        System.out.println("values:");
        IterativePolicyEvaluation.printValues(V, grid);
        System.out.println("policy:");
        IterativePolicyEvaluation.printPolicy(policy, grid);


    }
}
