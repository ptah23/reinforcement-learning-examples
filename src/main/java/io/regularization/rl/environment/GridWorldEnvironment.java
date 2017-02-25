package io.regularization.rl.environment;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static io.regularization.rl.environment.GridWorldAction.*;

/**
 * Created by ptah on 22/02/2017.
 */
public class GridWorldEnvironment {
    private int width;
    private int height;
    private GridWorldPosition currentPosition;
    private Map<GridWorldPosition, GridWorldReward> rewards = new HashMap<>();
    private Map<GridWorldPosition, List<GridWorldAction>> actions = new HashMap<>();

    public GridWorldEnvironment(int width, int height, GridWorldPosition startPosition) {
        this.width = width;
        this.height = height;
        currentPosition = startPosition;
    }

    public static GridWorldEnvironment standardGrid() {
        GridWorldEnvironment returnValue = new GridWorldEnvironment(3, 4,
                new GridWorldPosition(2, 0));
        returnValue.rewards.put(new GridWorldPosition(0, 3), new GridWorldReward(1));
        returnValue.rewards.put(new GridWorldPosition(1, 3), new GridWorldReward(-1));
        returnValue.actions.put(new GridWorldPosition(0, 0),
                Lists.newArrayList(DOWN, RIGHT));
        returnValue.actions.put(new GridWorldPosition(0, 1),
                Lists.newArrayList(LEFT, RIGHT));
        returnValue.actions.put(new GridWorldPosition(0, 2),
                Lists.newArrayList(LEFT, DOWN, RIGHT));
        returnValue.actions.put(new GridWorldPosition(1, 0),
                Lists.newArrayList(UP, DOWN));
        returnValue.actions.put(new GridWorldPosition(1, 2),
                Lists.newArrayList(UP, DOWN, RIGHT));
        returnValue.actions.put(new GridWorldPosition(2, 0),
                Lists.newArrayList(UP, RIGHT));
        returnValue.actions.put(new GridWorldPosition(2, 1),
                Lists.newArrayList(LEFT, RIGHT));
        returnValue.actions.put(new GridWorldPosition(2, 2),
                Lists.newArrayList(LEFT, RIGHT, UP));
        returnValue.actions.put(new GridWorldPosition(2, 3),
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
                ImmutableMap.<GridWorldPosition, GridWorldReward> builder()
                        .put(new GridWorldPosition(0, 0), new GridWorldReward(stepCost))
                        .put(new GridWorldPosition(0, 1), new GridWorldReward(stepCost))
                        .put(new GridWorldPosition(0, 2), new GridWorldReward(stepCost))
                        .put(new GridWorldPosition(1, 0), new GridWorldReward(stepCost))
                        .put(new GridWorldPosition(1, 2), new GridWorldReward(stepCost))
                        .put(new GridWorldPosition(2, 0), new GridWorldReward(stepCost))
                        .put(new GridWorldPosition(2, 1), new GridWorldReward(stepCost))
                        .put(new GridWorldPosition(2, 2), new GridWorldReward(stepCost))
                        .put(new GridWorldPosition(2, 3), new GridWorldReward(stepCost))
                        .build());
        return returnValue;
    }

    public boolean isTerminal(GridWorldPosition position) {
        return !actions.containsKey(position);
    }

    public GridWorldReward move(GridWorldAction action) {
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

    public Set<GridWorldPosition> allStates() {
        Set<GridWorldPosition> returnValue = Sets.union(rewards.keySet(), actions.keySet());
        return returnValue;
    }

    public boolean gameOver() {
        return !actions.containsKey(currentPosition);
    }

    public GridWorldPosition getCurrentPosition() {
        return currentPosition;
    }

    public void setCurrentPosition(GridWorldPosition currentPosition) {
        this.currentPosition = currentPosition;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public Map<GridWorldPosition,List<GridWorldAction>> getActions() {
        return actions;
    }

    public void setActions(Map<GridWorldPosition, List<GridWorldAction>> actions) {
        this.actions = actions;
    }

    public Map<GridWorldPosition, GridWorldReward> getRewards() {
        return rewards;
    }

    public void setRewards(Map<GridWorldPosition, GridWorldReward> rewards) {
        this.rewards = rewards;
    }
}
