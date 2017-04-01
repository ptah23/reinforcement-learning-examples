package io.regularization.rl.td0;

import io.regularization.rl.dynamicprogramming.IterativePolicyEvaluation;
import io.regularization.rl.environment.GridWorldAction;
import io.regularization.rl.environment.GridWorldEnvironment;
import io.regularization.rl.environment.GridWorldReward;
import io.regularization.rl.environment.GridWorldState;

import java.util.*;

/**
 * Created by ptah on 31/03/2017.
 */
public class SarsaControl {
    public static final float ALPHA = 0.1f;
    public static final float GAMMA = 0.9f;
    private static Random random = new Random();

    public static void main(String args[]) {
        GridWorldEnvironment grid = GridWorldEnvironment.negativeGrid(-0.1f);
        System.out.println("rewards");
        IterativePolicyEvaluation.printValues(grid.getRewards(), grid);
        Map<GridWorldState, Map<GridWorldAction, GridWorldReward>> Q = new HashMap<>();
        Map<GridWorldState, Integer> updateCounts = new HashMap<>();
        Map<GridWorldState, Map<GridWorldAction, Float>> updateCountsStateAction = new HashMap<>();
        for (GridWorldState state : grid.allStates()) {
            updateCounts.put(state, 0);
            Map<GridWorldAction, GridWorldReward> actionRewards = new HashMap<>();
            Map<GridWorldAction, Float> actionCounts = new HashMap<>();
            for (GridWorldAction action : GridWorldAction.values()) {
                actionRewards.put(action, new GridWorldReward(0.0f));
                actionCounts.put(action, 1.0f);
            }
            Q.put(state, actionRewards);
            updateCountsStateAction.put(state, actionCounts);
        }
        float t = 1.0f;
        List<Float> deltas = new ArrayList<>();
        for (int it = 0; it < 10000; it++) {
            if (it % 100 == 0) {
                t += 10e-3;
            }
            if (it % 2000 == 0) {
                System.out.println("it = " + it);
            }
            GridWorldState state = new GridWorldState(2, 0);
            grid.setCurrentPosition(state);
            GridWorldAction action = max(Q, state);
            action = nextAction(action, 0.5f / t);
            float biggestChange = 0f;
            while (!grid.gameOver()) {
                GridWorldReward reward = grid.move(action);
                GridWorldState state2 = grid.getCurrentPosition();
                GridWorldAction action2 = max(Q, state2);
                action2 = nextAction(action2, 0.5f / t);
                float alpha = ALPHA / updateCountsStateAction.get(state).get(action);
                updateCountsStateAction.get(state).put(action,
                        updateCountsStateAction.get(state).get(action) + 0.005f);
                GridWorldReward oldQsa = Q.get(state).get(action);
                Q.get(state).put(action, new GridWorldReward(Q.get(state).get(action).getValue() + alpha *
                        (reward.getValue() + GAMMA * Q.get(state2).get(action2).getValue()
                                - Q.get(state).get(action).getValue())));
                biggestChange = Math.max(biggestChange,
                        Math.abs(oldQsa.getValue() - Q.get(state).get(action).getValue()));
                updateCounts.put(state, updateCounts.get(state) + 1);
                state = state2;
                action = action2;
            }
            deltas.add(biggestChange);
        }
        Map<GridWorldState, GridWorldAction> policy = new HashMap<>();
        Map<GridWorldState, GridWorldReward> V = new HashMap<>();
        for (GridWorldState state : grid.getActions().keySet()) {
            GridWorldAction action = max(Q, state);
            policy.put(state, action);
            V.put(state, Q.get(state).get(action));
        }

        System.out.println("Values:");
        IterativePolicyEvaluation.printValues(V, grid);
        IterativePolicyEvaluation.printPolicy(policy, grid);


    }

    private static GridWorldAction max(Map<GridWorldState, Map<GridWorldAction, GridWorldReward>> q, GridWorldState state) {
        Optional<GridWorldAction> maxValue = q.get(state).entrySet().stream()
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
