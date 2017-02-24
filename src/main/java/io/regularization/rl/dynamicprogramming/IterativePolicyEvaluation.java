package io.regularization.rl.dynamicprogramming;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import io.regularization.rl.environment.GridWorldAction;
import io.regularization.rl.environment.GridWorldEnvironment;
import io.regularization.rl.environment.GridWorldPosition;
import io.regularization.rl.environment.GridWorldReward;

import java.util.Map;
import java.util.Random;

import static io.regularization.rl.environment.GridWorldAction.RIGHT;
import static io.regularization.rl.environment.GridWorldAction.UP;

/**
 * Created by ptah on 23/02/2017.
 */
public class IterativePolicyEvaluation {
    private static float SMALL_ENOUGH = 10e-4f, GAMMA = 0.9f;
    private static Random random = new Random();

    public static void printValues(Map<GridWorldPosition, GridWorldReward> V, GridWorldEnvironment grid) {
        for (int i = 0; i < grid.getWidth(); i++) {
            System.out.println("---------------------------");
            for (int j = 0; j < grid.getHeight(); j++) {
                double v = V.get(new GridWorldPosition(i, j)) != null ? V.get(new GridWorldPosition(i, j)).getValue() : 0;
                if (v >= 0) {
                    System.out.printf(" %.2f|", v);
                } else {
                    System.out.printf("%.2f|", v); //#-ve sign takes up an extra space
                }
            }
            System.out.print("\n");
        }
    }

    public static void printPolicy(Map<GridWorldPosition, GridWorldAction> P, GridWorldEnvironment grid) {
        System.out.println("Policy:");
        for (int i = 0; i < grid.getWidth(); i++) {
            System.out.println("---------------------------");
            for (int j = 0; j < grid.getHeight(); j++) {
                GridWorldAction a = P.get(new GridWorldPosition(i, j));
                System.out.printf(" %s |", a != null ? a.toString().charAt(0) : " ");
            }
            System.out.print("\n");
        }

    }

    public static void main(String args[]) {
        GridWorldEnvironment grid = GridWorldEnvironment.standardGrid();

        Map<GridWorldPosition, GridWorldReward> V = valueFunctionForRandomPolicy(grid);
        System.out.println("values for uniformly random actions:");
        printValues(V, grid);
        Map<GridWorldPosition, GridWorldAction> policy = ImmutableMap.<GridWorldPosition, GridWorldAction>builder()
                .put(new GridWorldPosition(2, 0), UP)
                .put(new GridWorldPosition(1, 0), UP)
                .put(new GridWorldPosition(0, 0), RIGHT)
                .put(new GridWorldPosition(0, 1), RIGHT)
                .put(new GridWorldPosition(0, 2), RIGHT)
                .put(new GridWorldPosition(1, 2), RIGHT)
                .put(new GridWorldPosition(2, 1), RIGHT)
                .put(new GridWorldPosition(2, 2), RIGHT)
                .put(new GridWorldPosition(2, 3), UP).build();
        printPolicy(policy, grid);
        V = valueFunctionForFixedPolicy(grid, policy, GAMMA);
        printValues(V, grid);

    }

    private static Map<GridWorldPosition, GridWorldReward> initialiseV(GridWorldEnvironment grid) {
        Map<GridWorldPosition, GridWorldReward> V = Maps.newHashMap();
        grid.allStates().stream().forEach(state -> V.put(state, new GridWorldReward(grid.getActions().containsKey(state) ? random.nextFloat() : 0.0f)));
        return V;
    }

    public static Map<GridWorldPosition, GridWorldReward> valueFunctionForRandomPolicy(GridWorldEnvironment grid) {
        float gamma = 1.0f;
        Map<GridWorldPosition, GridWorldReward> V = initialiseV(grid);
        while (true) {
            float biggestChange = 0;
            for (GridWorldPosition state : grid.allStates()) {
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

    public static Map<GridWorldPosition, GridWorldReward> valueFunctionForFixedPolicy(GridWorldEnvironment grid, Map<GridWorldPosition, GridWorldAction> policy, float gamma) {

        Map<GridWorldPosition, GridWorldReward> V = initialiseV(grid);
        while (true) {
            float biggestChange = 0;
            for (GridWorldPosition state : grid.allStates()) {
                float oldV = V.get(state).getValue();

                // V(state) only has value if it's not a terminal state
                if (policy.containsKey(state)) {
                    GridWorldAction action = policy.get(state);
                    grid.setCurrentPosition(state);
                    GridWorldReward r = grid.move(action);
                    V.put(state, new GridWorldReward(r.getValue() + gamma * V.get(grid.getCurrentPosition()).getValue()));
                    biggestChange = Math.max(biggestChange, Math.abs(oldV - V.get(state).getValue()));
                }
            }
            if (biggestChange < SMALL_ENOUGH) {
                break;
            }
        }
        return V;
    }


}
