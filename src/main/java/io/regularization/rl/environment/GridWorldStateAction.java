package io.regularization.rl.environment;

import java.util.Objects;

/**
 * Created by ptah on 11/03/2017.
 */
public class GridWorldStateAction {
    private GridWorldState state;
    private GridWorldAction action;

    public GridWorldStateAction(GridWorldState state, GridWorldAction action) {
        this.state = state;
        this.action = action;
    }

    public GridWorldState getState() {
        return state;
    }

    public GridWorldAction getAction() {
        return action;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        GridWorldStateAction that = (GridWorldStateAction) o;
        return Objects.equals(state, that.state) &&
                action == that.action;
    }

    @Override
    public int hashCode() {
        return Objects.hash(state, action);
    }
}
