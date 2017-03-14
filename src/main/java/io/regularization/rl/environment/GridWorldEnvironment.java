package io.regularization.rl.environment;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import java.util.*;

import static io.regularization.rl.environment.GridWorldAction.*;

/**
 * Created by ptah on 22/02/2017.
 */
public class GridWorldEnvironment {
    private int width;
    private int height;
    private GridWorldState currentPosition;
    private Map<GridWorldState, GridWorldReward> rewards = new HashMap<>();
    private Map<GridWorldState, List<GridWorldAction>> actions = new HashMap<>();
    private Set<GridWorldState> seen = new HashSet<>();

    public GridWorldEnvironment(int width, int height, GridWorldState startPosition) {
        this.width = width;
        this.height = height;
        currentPosition = startPosition;
    }

    public static GridWorldEnvironment standardGrid() {
        GridWorldEnvironment returnValue = new GridWorldEnvironment(3, 4,
                new GridWorldState(2, 0));
        returnValue.rewards.put(new GridWorldState(0, 3), new GridWorldReward(1));
        returnValue.rewards.put(new GridWorldState(1, 3), new GridWorldReward(-1));
        returnValue.actions.put(new GridWorldState(0, 0),
                Lists.newArrayList(DOWN, RIGHT));
        returnValue.actions.put(new GridWorldState(0, 1),
                Lists.newArrayList(LEFT, RIGHT));
        returnValue.actions.put(new GridWorldState(0, 2),
                Lists.newArrayList(LEFT, DOWN, RIGHT));
        returnValue.actions.put(new GridWorldState(1, 0),
                Lists.newArrayList(UP, DOWN));
        returnValue.actions.put(new GridWorldState(1, 2),
                Lists.newArrayList(UP, DOWN, RIGHT));
        returnValue.actions.put(new GridWorldState(2, 0),
                Lists.newArrayList(UP, RIGHT));
        returnValue.actions.put(new GridWorldState(2, 1),
                Lists.newArrayList(LEFT, RIGHT));
        returnValue.actions.put(new GridWorldState(2, 2),
                Lists.newArrayList(LEFT, RIGHT, UP));
        returnValue.actions.put(new GridWorldState(2, 3),
                Lists.newArrayList(LEFT, UP));
        return returnValue;
    }

    public static GridWorldEnvironment negativeGrid() {
        float stepCost = -0.1f;

        GridWorldEnvironment returnValue = negativeGrid(stepCost);
        return returnValue;
    }

    public static GridWorldEnvironment negativeGrid(float stepCost) {
        GridWorldEnvironment returnValue = standardGrid();

        returnValue.rewards.putAll(
                ImmutableMap.<GridWorldState, GridWorldReward>builder()
                        .put(new GridWorldState(0, 0), new GridWorldReward(stepCost))
                        .put(new GridWorldState(0, 1), new GridWorldReward(stepCost))
                        .put(new GridWorldState(0, 2), new GridWorldReward(stepCost))
                        .put(new GridWorldState(1, 0), new GridWorldReward(stepCost))
                        .put(new GridWorldState(1, 2), new GridWorldReward(stepCost))
                        .put(new GridWorldState(2, 0), new GridWorldReward(stepCost))
                        .put(new GridWorldState(2, 1), new GridWorldReward(stepCost))
                        .put(new GridWorldState(2, 2), new GridWorldReward(stepCost))
                        .put(new GridWorldState(2, 3), new GridWorldReward(stepCost))
                        .build());
        return returnValue;
    }

    public boolean isTerminal(GridWorldState position) {
        return !actions.containsKey(position);
    }


    public GridWorldReward move(GridWorldAction action) {
        seen.add(currentPosition);
        if (actions.get(currentPosition).contains(action)) {
            currentPosition = action.perform(currentPosition);
        }

        return rewards.get(currentPosition) != null ? rewards.get(currentPosition) : new GridWorldReward(0.0f);
    }

    public GridWorldReward undoMove(GridWorldAction action) {
        currentPosition = action.undo(currentPosition);
        assert (allStates().contains(currentPosition));
        return rewards.get(currentPosition);
    }

    public Set<GridWorldState> allStates() {
        Set<GridWorldState> returnValue = Sets.union(rewards.keySet(), actions.keySet());
        return returnValue;
    }

    public boolean gameOver() {
        return !actions.containsKey(currentPosition);
    }

    public GridWorldState getCurrentPosition() {
        return currentPosition;
    }

    public void setCurrentPosition(GridWorldState currentPosition) {
        this.currentPosition = currentPosition;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public Map<GridWorldState, List<GridWorldAction>> getActions() {
        return actions;
    }

    public void setActions(Map<GridWorldState, List<GridWorldAction>> actions) {
        this.actions = actions;
    }

    public Map<GridWorldState, GridWorldReward> getRewards() {
        return rewards;
    }

    public void setRewards(Map<GridWorldState, GridWorldReward> rewards) {
        this.rewards = rewards;
    }

    public boolean hasSeen(GridWorldState state) {
        return seen.contains(state);
    }

    public void clearSeen() {
        seen.clear();
    }
}
