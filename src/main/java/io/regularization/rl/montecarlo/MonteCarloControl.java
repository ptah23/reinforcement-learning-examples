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
public class MonteCarloControl {
    static Random random = new Random();
    private static float GAMMA = 0.9f;

    public static void main(String args[]) {
        GridWorldEnvironment grid = GridWorldEnvironment.negativeGrid();
        System.out.println("rewards:");
        IterativePolicyEvaluation.printValues(grid.getRewards(), grid);
        //random policy
        Map<GridWorldState, GridWorldAction> policy = new HashMap<>();
        for (GridWorldState current : grid.getActions().keySet()) {
            policy.put(current, grid.getActions().get(current).get(random.nextInt(grid.getActions().get(current).size())));
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
        for (int t = 0; t < 20000; t++) {
            //if( t % 10 == 0) {
            System.out.println(t);
            //}
            float biggestChange = 0.0f;
            Map<GridWorldStateAction, GridWorldReward> stateActionsReturns = playGameRandomStart(grid, policy);
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

    private static Map<GridWorldStateAction, GridWorldReward> playGameRandomStart(GridWorldEnvironment grid, Map<GridWorldState, GridWorldAction> policy) {

        List<GridWorldState> startStates = new ArrayList<>(grid.getActions().keySet());
        int startIndex = random.nextInt(startStates.size());
        GridWorldState state = startStates.get(startIndex);
        grid.setCurrentPosition(state);
        LinkedHashMap<GridWorldStateAction, GridWorldReward> returnValue = playGame(grid, policy);

        return returnValue;
    }

    private static LinkedHashMap<GridWorldStateAction, GridWorldReward> playGame(GridWorldEnvironment grid, Map<GridWorldState, GridWorldAction> policy) {
        GridWorldState state = grid.getCurrentPosition();
        grid.clearSeen();
        LinkedHashMap<GridWorldStateAction, GridWorldReward> statesAndRewards = Maps.newLinkedHashMap();
        GridWorldAction action = grid.getActions().get(state).get(random.nextInt(grid.getActions().get(state).size()));
        // statesAndRewards.put(new GridWorldStateAction(state, action), grid.getRewards().get(state));
        while (action != null) {
            action = playRound(grid, policy, random, statesAndRewards);
        }
        LinkedHashMap<GridWorldStateAction, GridWorldReward> returnValue = calculateG(statesAndRewards);
        return returnValue;
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

    public static GridWorldAction playRound(GridWorldEnvironment grid, Map<GridWorldState, GridWorldAction> policy, Random random, LinkedHashMap<GridWorldStateAction, GridWorldReward> statesAndRewards) {
        GridWorldState state = grid.getCurrentPosition();
        GridWorldAction action = policy.get(state);
        GridWorldState oldState = grid.getCurrentPosition();
        GridWorldReward reward = grid.move(action);
        state = grid.getCurrentPosition();
        if (state.equals(oldState)) {
            reward = new GridWorldReward(-100f);
            statesAndRewards.put(new GridWorldStateAction(state, null), reward);
            action = null;
        } else if (grid.hasSeen(state)) {
            //cycle
            reward = new GridWorldReward(-100f);
            statesAndRewards.put(new GridWorldStateAction(state, null), reward);
            action = null;
        } else if (grid.gameOver()) {
            statesAndRewards.put(new GridWorldStateAction(state, null), reward);
            action = null;
        } else {
            action = policy.get(state);
            statesAndRewards.put(new GridWorldStateAction(state, action), reward);

        }
        return action;
    }
}
