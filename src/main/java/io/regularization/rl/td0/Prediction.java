package io.regularization.rl.td0;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import io.regularization.rl.dynamicprogramming.IterativePolicyEvaluation;
import io.regularization.rl.environment.GridWorldAction;
import io.regularization.rl.environment.GridWorldEnvironment;
import io.regularization.rl.environment.GridWorldReward;
import io.regularization.rl.environment.GridWorldState;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static io.regularization.rl.environment.GridWorldAction.RIGHT;
import static io.regularization.rl.environment.GridWorldAction.UP;
import static io.regularization.rl.montecarlo.MonteCarloControlNoExploringStarts.nextAction;

/**
 * Created by ptah on 30/03/2017.
 */
public class Prediction {

    private static float GAMMA = 0.9f;
    private static float ALPHA = 0.1f;

    public static void main(String args[]) {
        GridWorldEnvironment grid = GridWorldEnvironment.standardGrid();
        System.out.println("Rewards:");
        IterativePolicyEvaluation.printValues(grid.getRewards(), grid);
        Map<GridWorldState, GridWorldAction> policy = ImmutableMap.<GridWorldState, GridWorldAction>builder()
                .put(new GridWorldState(2, 0), UP)
                .put(new GridWorldState(1, 0), UP)
                .put(new GridWorldState(0, 0), RIGHT)
                .put(new GridWorldState(0, 1), RIGHT)
                .put(new GridWorldState(0, 2), RIGHT)
                .put(new GridWorldState(1, 2), RIGHT)
                .put(new GridWorldState(2, 1), RIGHT)
                .put(new GridWorldState(2, 2), RIGHT)
                .put(new GridWorldState(2, 3), UP)
                .build();
        Map<GridWorldState, GridWorldReward> V = Maps.newHashMap();
        for (GridWorldState state : grid.allStates()) {
            V.put(state, new GridWorldReward(0.0f));
        }
        for (int i = 0; i < 1000; i++) {
            LinkedHashMap<GridWorldState, GridWorldReward> statesAndRewards = playGame(grid, policy);
            List<GridWorldState> list = Lists.reverse(new ArrayList<GridWorldState>(statesAndRewards.keySet()));
            GridWorldState s2 = list.get(0);
            for (int t = 1; t < list.size(); t++) {
                GridWorldState s = list.get(t);
                V.put(s, new GridWorldReward(V.get(s).getValue() + ALPHA * (
                        statesAndRewards.get(s2).getValue() + GAMMA * V.get(s2).getValue() - V.get(s).getValue())));
                s2 = s;
            }

        }
        System.out.println("Values:");
        IterativePolicyEvaluation.printValues(V, grid);
        System.out.println("policy:");
        IterativePolicyEvaluation.printPolicy(policy, grid);
    }

    private static LinkedHashMap<GridWorldState, GridWorldReward> playGame(GridWorldEnvironment grid, Map<GridWorldState, GridWorldAction> policy) {
        LinkedHashMap<GridWorldState, GridWorldReward> returnValue = Maps.newLinkedHashMap();
        GridWorldState state = new GridWorldState(2, 0);
        grid.setCurrentPosition(state);
        returnValue.put(state, new GridWorldReward(0.0f));
        while (!grid.gameOver()) {
            GridWorldAction action = nextAction(grid, policy, state);
            GridWorldReward reward = grid.move(action);
            state = grid.getCurrentPosition();
            returnValue.put(state, reward);
        }
        return returnValue;
    }
}
