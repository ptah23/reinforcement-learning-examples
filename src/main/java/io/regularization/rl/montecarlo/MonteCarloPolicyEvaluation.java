package io.regularization.rl.montecarlo;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import io.regularization.rl.dynamicprogramming.IterativePolicyEvaluation;
import io.regularization.rl.environment.GridWorldAction;
import io.regularization.rl.environment.GridWorldEnvironment;
import io.regularization.rl.environment.GridWorldPosition;
import io.regularization.rl.environment.GridWorldReward;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static io.regularization.rl.environment.GridWorldAction.RIGHT;
import static io.regularization.rl.environment.GridWorldAction.UP;

/**
 * Created by ptah on 02/03/2017.
 */
public class MonteCarloPolicyEvaluation {
    private static float GAMMA = 0.9f;

    public static void main(String args[]) {
        GridWorldEnvironment grid = GridWorldEnvironment.standardGrid();
        IterativePolicyEvaluation.printValues(grid.getRewards(), grid);
        Map<GridWorldPosition, GridWorldAction> policy = ImmutableMap.<GridWorldPosition, GridWorldAction>builder()
                .put(new GridWorldPosition(2, 0), UP)
                .put(new GridWorldPosition(1, 0), UP)
                .put(new GridWorldPosition(0, 0), RIGHT)
                .put(new GridWorldPosition(0, 1), RIGHT)
                .put(new GridWorldPosition(0, 2), RIGHT)
                .put(new GridWorldPosition(1, 2), RIGHT)
                .put(new GridWorldPosition(2, 1), RIGHT)
                .put(new GridWorldPosition(2, 2), RIGHT)
                .put(new GridWorldPosition(2, 3), UP)
                .build();
        Map<GridWorldPosition, GridWorldReward> V = Maps.newHashMap();
        Map<GridWorldPosition, ArrayList<Float>> returns = Maps.newHashMap();
        for (GridWorldPosition position : grid.allStates()) {
            if (grid.getActions().containsKey(position)) {
                returns.put(position, new ArrayList<>());
            } else {
                V.put(position, new GridWorldReward(0.0f));
            }
        }
        for (int t = 0; t < 1000; t++) {
            Map<GridWorldPosition, GridWorldReward> statesAndReturns = playGame(grid, policy);
            Set<GridWorldPosition> seenStates = Sets.newHashSet();
            ListIterator<GridWorldPosition> iterator = new ArrayList(statesAndReturns.keySet()).listIterator(
                    statesAndReturns.size());
            while (iterator.hasPrevious()) {
                GridWorldPosition key = iterator.previous();

                //first-visit evaluation
                if (!seenStates.contains(key)) {
                    ArrayList<Float> list = returns.get(key);
                    list.add(statesAndReturns.get(key).getValue());
                    V.put(key, new GridWorldReward(
                            Nd4j.mean(Nd4j.create(list.stream().mapToDouble(f -> f != null ? f : Float.NaN) // Or whatever default you want.
                                    .toArray())).getFloat(0)));
                    seenStates.add(key);
                }
            }
        }
        IterativePolicyEvaluation.printValues(V, grid);
        IterativePolicyEvaluation.printPolicy(policy, grid);

    }

    private static Map<GridWorldPosition, GridWorldReward> playGame(GridWorldEnvironment grid, Map<GridWorldPosition, GridWorldAction> policy) {
        Random random = new Random();
        List<GridWorldPosition> startStates = new ArrayList(grid.getActions().keySet());
        int startIndex = random.nextInt(startStates.size());
        GridWorldPosition state = startStates.get(startIndex);
        grid.setCurrentPosition(state);
        LinkedHashMap<GridWorldPosition, GridWorldReward> statesAndRewards = Maps.newLinkedHashMap();
        statesAndRewards.put(state, new GridWorldReward(0.0f));
        while (!grid.gameOver()) {
            GridWorldAction action = policy.get(state);
            GridWorldReward reward = grid.move(action);
            state = grid.getCurrentPosition();
            statesAndRewards.put(state, reward);
        }
        float G = 0;
        LinkedHashMap<GridWorldPosition, GridWorldReward> returnValue = Maps.newLinkedHashMap();
        boolean first = true;
        ListIterator<GridWorldPosition> iterator = new ArrayList(statesAndRewards.keySet()).listIterator(statesAndRewards.size());
        while (iterator.hasPrevious()) {
            GridWorldPosition key = iterator.previous();
            if (first) {
                first = false;
            } else {
                returnValue.put(key, new GridWorldReward(G));
            }
            G = statesAndRewards.get(key).getValue() + GAMMA * G;
        }

        return returnValue;

    }
}
