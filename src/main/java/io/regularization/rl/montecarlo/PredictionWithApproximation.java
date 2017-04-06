package io.regularization.rl.montecarlo;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import io.regularization.rl.approximators.Model;
import io.regularization.rl.dynamicprogramming.IterativePolicyEvaluation;
import io.regularization.rl.environment.GridWorldAction;
import io.regularization.rl.environment.GridWorldEnvironment;
import io.regularization.rl.environment.GridWorldReward;
import io.regularization.rl.environment.GridWorldState;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

import static io.regularization.rl.environment.GridWorldAction.*;

/**
 * Created by ptah on 02/03/2017.
 */

public class PredictionWithApproximation {
    private static final float LEARNING_RATE = 0.001f;
    private static float GAMMA = 0.9f;
    private static boolean windy = false;
    private static Random random = new Random();


    public static void main(String args[]) {
        GridWorldEnvironment grid = GridWorldEnvironment.standardGrid();
        IterativePolicyEvaluation.printValues(grid.getRewards(), grid);
        Map<GridWorldState, GridWorldAction> policy = windy ? ImmutableMap.<GridWorldState, GridWorldAction>builder()
                .put(new GridWorldState(2, 0), UP)
                .put(new GridWorldState(1, 0), UP)
                .put(new GridWorldState(0, 0), RIGHT)
                .put(new GridWorldState(0, 1), RIGHT)
                .put(new GridWorldState(0, 2), RIGHT)
                .put(new GridWorldState(1, 2), UP)
                .put(new GridWorldState(2, 1), LEFT)
                .put(new GridWorldState(2, 2), UP)
                .put(new GridWorldState(2, 3), LEFT)
                .build()
                :
                ImmutableMap.<GridWorldState, GridWorldAction>builder()
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

        Model model = new Model();
        float t = 1.0f;
        for (int it = 0; it < 20000; it++) {
            if (it % 100 == 0) {
                t += 0.01f;
            }
            float alpha = LEARNING_RATE / t;
            Map<GridWorldState, GridWorldReward> statesAndReturns = playGame(grid, policy);
            Set<GridWorldState> seenStates = Sets.newHashSet();
            ListIterator<GridWorldState> iterator = new ArrayList(statesAndReturns.keySet()).listIterator(
                    statesAndReturns.size());
            while (iterator.hasPrevious()) {
                GridWorldState key = iterator.previous();

                //first-visit evaluation
                if (!seenStates.contains(key)) {
                    INDArray grad = model.grad(key).transpose().mul(alpha * (statesAndReturns.get(key).getValue()
                            - model.predict(key)));
                    model.setTheta(model.getTheta().add(grad));
                    seenStates.add(key);
                }
            }
        }
        Map<GridWorldState, GridWorldReward> V = Maps.newHashMap();
        for (GridWorldState position : grid.allStates()) {
            if (grid.getActions().containsKey(position)) {
                V.put(position, new GridWorldReward(model.getTheta().mmul(model.s2x(position)).getFloat(0)));
            } else {
                V.put(position, new GridWorldReward(0.0f));
            }
        }
        IterativePolicyEvaluation.printValues(V, grid);
        IterativePolicyEvaluation.printPolicy(policy, grid);

    }

    private static GridWorldAction randomAction(GridWorldAction action) {
        double probability = random.nextGaussian();
        if (probability < 0.5f) {
            return action;
        } else {
            List<GridWorldAction> list = new ArrayList(Arrays.asList(GridWorldAction.values()));
            list.remove(action);
            return list.get(random.nextInt(list.size()));
        }
    }

    private static Map<GridWorldState, GridWorldReward> playGame(GridWorldEnvironment grid, Map<GridWorldState, GridWorldAction> policy) {
        Random random = new Random();
        List<GridWorldState> startStates = new ArrayList<>(grid.getActions().keySet());
        int startIndex = random.nextInt(startStates.size());
        GridWorldState state = startStates.get(startIndex);
        grid.setCurrentPosition(state);
        LinkedHashMap<GridWorldState, GridWorldReward> statesAndRewards = Maps.newLinkedHashMap();
        statesAndRewards.put(state, new GridWorldReward(0.0f));
        while (!grid.gameOver()) {
            GridWorldAction action = policy.get(state);
            if (windy) {
                action = randomAction(action);
            }
            GridWorldReward reward = grid.move(action);
            state = grid.getCurrentPosition();
            statesAndRewards.put(state, reward);
        }
        float G = 0;
        LinkedHashMap<GridWorldState, GridWorldReward> returnValue = Maps.newLinkedHashMap();
        boolean first = true;
        ListIterator<GridWorldState> iterator = new ArrayList(statesAndRewards.keySet()).listIterator(statesAndRewards.size());
        while (iterator.hasPrevious()) {
            GridWorldState key = iterator.previous();
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
