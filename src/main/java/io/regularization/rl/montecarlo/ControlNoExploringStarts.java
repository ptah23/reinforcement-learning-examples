package io.regularization.rl.montecarlo;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import io.regularization.rl.dynamicprogramming.IterativePolicyEvaluation;
import io.regularization.rl.environment.*;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Created by ptah on 09/03/2017.
 */
public class ControlNoExploringStarts {
    static Random random = new Random();
    private static float GAMMA = 0.9f;

    public static void main(String args[]) {
        GridWorldEnvironment grid = GridWorldEnvironment.negativeGrid();
        System.out.println("rewards:");
        IterativePolicyEvaluation.printValues(grid.getRewards(), grid);
        //random policy
        Map<GridWorldState, GridWorldAction> policy = new HashMap<>();
        for (GridWorldState current : grid.getActions().keySet()) {
            policy.put(current, randomAction(grid, current));
        }
        System.out.println("random policy:");
        IterativePolicyEvaluation.printPolicy(policy, grid);
        Map<GridWorldState, Map<GridWorldAction, GridWorldReward>> Q = new HashMap<>();
        Map<GridWorldStateAction, List<GridWorldReward>> returns = new HashMap<>();

        //initialize
        for (GridWorldState state : grid.allStates()) {
            if (grid.getActions().containsKey(state)) {
                Map<GridWorldAction, GridWorldReward> actionReward = new HashMap<>();
                for (GridWorldAction action : grid.getActions().get(state)) {
                    actionReward.put(action, new GridWorldReward(0.0f));
                    returns.put(new GridWorldStateAction(state, action),
                            Lists.newArrayList());
                }
                Q.put(state, actionReward);
            }
        }
        //repeat until convergence
        List<Float> deltas = new ArrayList<>();
        for (int t = 0; t < 5000; t++) {
            //if( t % 10 == 0) {
            System.out.println(t);
            //}
            float biggestChange = 0.0f;
            Map<GridWorldStateAction, GridWorldReward> stateActionsReturns = playGameStart(grid, policy);
            ListIterator<GridWorldStateAction> iterator = new ArrayList(stateActionsReturns.keySet()).listIterator(stateActionsReturns.size());
            Set<GridWorldStateAction> seenStates = new HashSet<>();
            while (iterator.hasPrevious()) {
                GridWorldStateAction key = iterator.previous();
                if (!seenStates.contains(key)) {
                    float oldQ = Q.get(key.getState()).get(key.getAction()).getValue();
                    returns.get(key).add(stateActionsReturns.get(key));
                    float newQ = Nd4j.mean(Nd4j.create(returns.get(key)
                            .stream().mapToDouble(f -> f != null ? f.getValue() : Float.NaN) // Or whatever default you want.
                            .toArray())).getFloat(0);
                    Q.get(key.getState()).put(key.getAction(),
                            new GridWorldReward(newQ));
                    biggestChange = Math.max(biggestChange, Math.abs(oldQ - newQ));
                    seenStates.add(key);
                }
            }
            deltas.add(biggestChange);
            for (Map.Entry<GridWorldState, GridWorldAction> stateAction : policy.entrySet()) {
                Optional<GridWorldAction> maxValue = Q.get(stateAction.getKey()).entrySet().stream()
                        .max(
                                Comparator.comparing(
                                        entry -> entry.getValue().getValue()
                                )
                        ).map(
                                maxResult -> maxResult.getKey()
                        );
                if (maxValue.isPresent()) {
                    policy.put(stateAction.getKey(), maxValue.get());
                }
            }
            IterativePolicyEvaluation.printPolicy(policy, grid);


        }
        System.out.println("final policy:");
        IterativePolicyEvaluation.printPolicy(policy, grid);
        Map<GridWorldState, GridWorldReward> V = new HashMap<>();
        for (Map.Entry<GridWorldState, Map<GridWorldAction, GridWorldReward>> entry : Q.entrySet()) {
            V.put(entry.getKey(), entry.getValue().entrySet()
                    .stream()
                    .max(Comparator.comparing(
                            rewardEntry -> rewardEntry.getValue().getValue()))
                    .get().getValue());
        }
        System.out.println("final values:");
        IterativePolicyEvaluation.printValues(V, grid);
    }

    private static Map<GridWorldStateAction, GridWorldReward> playGameStart(GridWorldEnvironment grid, Map<GridWorldState, GridWorldAction> policy) {

        List<GridWorldState> startStates = new ArrayList<>(grid.getActions().keySet());
        int startIndex = 2;
        GridWorldState state = startStates.get(startIndex);
        grid.setCurrentPosition(state);
        LinkedHashMap<GridWorldStateAction, GridWorldReward> returnValue = playGame(grid, policy);

        return returnValue;
    }

    private static LinkedHashMap<GridWorldStateAction, GridWorldReward> playGame(GridWorldEnvironment grid, Map<GridWorldState, GridWorldAction> policy) {
        GridWorldState state = grid.getCurrentPosition();
        grid.clearSeen();
        LinkedHashMap<GridWorldStateAction, GridWorldReward> statesAndRewards = Maps.newLinkedHashMap();
        GridWorldAction action = randomAction(grid, state);
        statesAndRewards.put(new GridWorldStateAction(state, action), grid.getRewards().get(state));
        action = nextAction(grid, policy, state);
        while (action != null) {
            action = playRound(action, grid, policy, statesAndRewards);
        }
        LinkedHashMap<GridWorldStateAction, GridWorldReward> returnValue = calculateG(statesAndRewards);
        return returnValue;
    }

    private static GridWorldAction randomAction(GridWorldEnvironment grid, GridWorldState state) {
        return grid.getActions().get(state).get(random.nextInt(grid.getActions().get(state).size()));
    }

    public static LinkedHashMap<GridWorldStateAction, GridWorldReward> calculateG(LinkedHashMap<GridWorldStateAction, GridWorldReward> statesAndRewards) {
        float G = 0;
        LinkedHashMap<GridWorldStateAction, GridWorldReward> returnValue = Maps.newLinkedHashMap();
        boolean first = true;
        ListIterator<GridWorldStateAction> iterator = new ArrayList(statesAndRewards.keySet()).listIterator(statesAndRewards.size());
        while (iterator.hasPrevious()) {
            GridWorldStateAction key = iterator.previous();
            if (first) {
                first = false;
            } else {
                returnValue.put(key, new GridWorldReward(G));
            }
            G = statesAndRewards.get(key).getValue() + (GAMMA * G);
        }
        return returnValue;
    }

    public static GridWorldAction playRound(GridWorldAction action, GridWorldEnvironment grid, Map<GridWorldState,
            GridWorldAction> policy, LinkedHashMap<GridWorldStateAction, GridWorldReward> statesAndRewards) {
        GridWorldReward reward = grid.move(action);
        GridWorldState state = grid.getCurrentPosition();
        if (grid.gameOver()) {
            statesAndRewards.put(new GridWorldStateAction(state, null), reward);
            action = null;
        } else {
            action = nextAction(grid, policy, state);
            statesAndRewards.put(new GridWorldStateAction(state, action), reward);

        }
        return action;
    }

    public static GridWorldAction nextAction(GridWorldEnvironment grid, Map<GridWorldState, GridWorldAction> policy,
                                             GridWorldState state) {
        float eps = 0.1f;
        float p = random.nextFloat();
        if (p < (1 - eps)) {
            return policy.get(state);
        } else {
            return randomAction(grid, state);
        }
    }
}
