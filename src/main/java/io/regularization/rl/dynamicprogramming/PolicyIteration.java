package io.regularization.rl.dynamicprogramming;

import com.google.common.collect.Maps;
import io.regularization.rl.environment.GridWorldAction;
import io.regularization.rl.environment.GridWorldEnvironment;
import io.regularization.rl.environment.GridWorldPosition;
import io.regularization.rl.environment.GridWorldReward;

import java.util.Map;
import java.util.Random;

/**
 * Created by ptah on 24/02/2017.
 */
public class PolicyIteration {
    private static final GridWorldAction[] ALL_POSSIBLE_ACTIONS = GridWorldAction.values();
    private static float SMALL_ENOUGH = 10e-4f, GAMMA = 0.9f;
    //private static

    public static void main(String args[]) {
        Random random = new Random();
        GridWorldEnvironment grid = GridWorldEnvironment.negativeGrid();
        System.out.println("rewards:");
        IterativePolicyEvaluation.printValues(grid.getRewards(), grid);
        //randomly choose action and update as we learn
        Map<GridWorldPosition, GridWorldAction> policy = Maps.newHashMap();
        for (GridWorldPosition state : grid.getActions().keySet()) {
            policy.put(state, ALL_POSSIBLE_ACTIONS[random.nextInt(ALL_POSSIBLE_ACTIONS.length)]);
        }
        System.out.println("initial policy");
        IterativePolicyEvaluation.printPolicy(policy, grid);
        boolean isPolicyConverged = false;
        Map<GridWorldPosition, GridWorldReward> V = null;
        while (!isPolicyConverged) {
            V = IterativePolicyEvaluation.valueFunctionForFixedPolicy(grid, policy, GAMMA);
            isPolicyConverged = improvePolicy(grid, policy, V);
        }
        System.out.println("values:");
        IterativePolicyEvaluation.printValues(V, grid);
        System.out.println("policy:");
        IterativePolicyEvaluation.printPolicy(policy, grid);
    }

    private static boolean improvePolicy(GridWorldEnvironment grid, Map<GridWorldPosition, GridWorldAction> policy, Map<GridWorldPosition, GridWorldReward> V) {
        boolean returnValue = true;
        for (GridWorldPosition state : grid.allStates()) {
            if (policy.containsKey(state)) {
                GridWorldAction oldAction = policy.get(state);
                float bestValue = Float.NEGATIVE_INFINITY;
                GridWorldAction newAction = null;
                for (GridWorldAction action : ALL_POSSIBLE_ACTIONS) {
                    grid.setCurrentPosition(state);
                    GridWorldReward reward = grid.move(action);
                    float v = reward.getValue() + GAMMA * V.get(grid.getCurrentPosition()).getValue();
                    if (v > bestValue) {
                        bestValue = v;
                        newAction = action;
                    }

                }
                policy.put(state, newAction);
                if (newAction != oldAction) {
                    returnValue = false;
                }
            }

        }

        return returnValue;
    }
}
