package io.regularization.rl.dynamicprogramming;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import io.regularization.rl.environment.GridWorldAction;
import io.regularization.rl.environment.GridWorldEnvironment;
import io.regularization.rl.environment.GridWorldReward;
import io.regularization.rl.environment.GridWorldState;

import java.util.Map;
import java.util.Random;

import static io.regularization.rl.environment.GridWorldAction.RIGHT;
import static io.regularization.rl.environment.GridWorldAction.UP;

/**
 * Created by ptah on 23/02/2017.
 */
public class IterativePolicyEvaluation {
    private static float SMALL_ENOUGH = 10e-2f, GAMMA = 0.9f;
    private static Random random = new Random();

    public static void printValues(Map<GridWorldState, GridWorldReward> V, GridWorldEnvironment grid) {
        for (int i = 0; i < grid.getWidth(); i++) {
            System.out.println("---------------------------");
            for (int j = 0; j < grid.getHeight(); j++) {
                double v = V.get(new GridWorldState(i, j)) != null ? V.get(new GridWorldState(i, j)).getValue() : 0;
                if (v >= 0) {
                    System.out.printf(" %.2f|", v);
                } else {
                    System.out.printf("%.2f|", v); //#-ve sign takes up an extra space
                }
            }
            System.out.print("\n");
        }
    }

    public static void printPolicy(Map<GridWorldState, GridWorldAction> P, GridWorldEnvironment grid) {
        System.out.println("Policy:");
        for (int i = 0; i < grid.getWidth(); i++) {
            System.out.println("---------------------------");
            for (int j = 0; j < grid.getHeight(); j++) {
                GridWorldAction a = P.get(new GridWorldState(i, j));
                System.out.printf(" %s |", a != null ? a.toString().charAt(0) : " ");
            }
            System.out.print("\n");
        }

    }

    public static void main(String args[]) {
        GridWorldEnvironment grid = GridWorldEnvironment.standardGrid();

        Map<GridWorldState, GridWorldReward> V = valueFunctionForRandomPolicy(grid);
        System.out.println("values for uniformly random actions:");
        printValues(V, grid);
        Map<GridWorldState, GridWorldAction> policy = ImmutableMap.<GridWorldState, GridWorldAction>builder()
                .put(new GridWorldState(2, 0), UP)
                .put(new GridWorldState(1, 0), UP)
                .put(new GridWorldState(0, 0), RIGHT)
                .put(new GridWorldState(0, 1), RIGHT)
                .put(new GridWorldState(0, 2), RIGHT)
                .put(new GridWorldState(1, 2), RIGHT)
                .put(new GridWorldState(2, 1), RIGHT)
                .put(new GridWorldState(2, 2), RIGHT)
                .put(new GridWorldState(2, 3), UP).build();
        printPolicy(policy, grid);
        V = valueFunctionForFixedPolicy(grid, policy, GAMMA, false);
        printValues(V, grid);

    }

    public static Map<GridWorldState, GridWorldReward> initialiseV(GridWorldEnvironment grid) {
        Map<GridWorldState, GridWorldReward> V = Maps.newHashMap();
        grid.allStates().stream().forEach(state -> V.put(state, new GridWorldReward(grid.getActions().containsKey(state) ? random.nextFloat() : 0.0f)));
        return V;
    }

    public static Map<GridWorldState, GridWorldReward> valueFunctionForRandomPolicy(GridWorldEnvironment grid) {
        float gamma = 1.0f;
        Map<GridWorldState, GridWorldReward> V = initialiseV(grid);
        while (true) {
            float biggestChange = 0;
            for (GridWorldState state : grid.allStates()) {
                float oldV = V.get(state).getValue();

                // V(state) only has value if it's not a terminal state
                if (grid.getActions().containsKey(state)) {

                    float newV = 0; //we will accumulate the answer
                    float pA = 1.0f / grid.getActions().get(state).size();//each action has equal probability
                    for (GridWorldAction action : grid.getActions().get(state)) {
                        grid.setCurrentPosition(state);
                        GridWorldReward r = grid.move(action);
                        System.out.println("currentposition:" + grid.getCurrentPosition() + ", action:" + action + ", reward:" + r);
                        newV += pA * (r.getValue() + gamma * V.get(grid.getCurrentPosition()).getValue());
                    }
                    V.put(state, new GridWorldReward(newV));
                    biggestChange = Math.max(biggestChange, Math.abs(oldV - V.get(state).getValue()));
                }
            }
            if (biggestChange < SMALL_ENOUGH) {
                break;
            }
        }
        return V;
    }

    public static Map<GridWorldState, GridWorldReward> valueFunctionForFixedPolicy(GridWorldEnvironment grid,
                                                                                   Map<GridWorldState,
                                                                                              GridWorldAction> policy,
                                                                                   float gamma, boolean windy) {

        Map<GridWorldState, GridWorldReward> V = initialiseV(grid);
        while (true) {
            float biggestChange = 0;
            for (GridWorldState state : grid.allStates()) {
                float oldV = V.get(state).getValue();

                // V(state) only has value if it's not a terminal state
                if (policy.containsKey(state)) {
                    float newV = 0.0f;
                    if (windy) {
                        newV = calculateVRandom(grid, gamma, V, state, policy.get(state));
                    } else {
                        newV = calculateVdeterministic(grid, gamma, V, state, policy.get(state));
                    }
                    V.put(state, new GridWorldReward(newV));
                    biggestChange = Math.max(biggestChange, Math.abs(oldV - V.get(state).getValue()));
                }
            }
            if (biggestChange < SMALL_ENOUGH) {
                break;
            }
        }
        return V;
    }

    public static float calculateVdeterministic(GridWorldEnvironment grid, float gamma, Map<GridWorldState,
            GridWorldReward> v, GridWorldState state, GridWorldAction action) {
        grid.setCurrentPosition(state);
        GridWorldReward r = grid.move(action);
        return r.getValue() + gamma * v.get(grid.getCurrentPosition()).getValue();
    }

    public static float calculateVRandom(GridWorldEnvironment grid, float gamma, Map<GridWorldState, GridWorldReward> V, GridWorldState state, GridWorldAction chosenAction) {
        float returnValue = 0.0f;
        for (GridWorldAction resultingAction : GridWorldAction.values()) {// resulting action
            float p = 0.0f;
            if (chosenAction == resultingAction) {
                p = 0.5f;
            } else {
                p = 0.5f / 3;
            }
            grid.setCurrentPosition(state);
            GridWorldReward reward = grid.move(resultingAction);
            returnValue += p * (reward.getValue() + gamma * V.get(grid.getCurrentPosition()).getValue());
        }
        return returnValue;
    }



}
