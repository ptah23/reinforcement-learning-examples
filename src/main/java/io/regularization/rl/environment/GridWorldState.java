package io.regularization.rl.environment;

import com.google.common.base.MoreObjects;

import java.util.Objects;

/**
 * Created by ptah on 22/02/2017.
 */
public class GridWorldState {
    private int i;
    private int j;

    public GridWorldState(int i, int j) {
        this.i = i;
        this.j = j;
    }

    public int getI() {
        return i;
    }

    public int getJ() {
        return j;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        GridWorldState that = (GridWorldState) o;
        return i == that.i &&
                j == that.j;
    }

    @Override
    public int hashCode() {
        return Objects.hash(i, j);
    }

    @Override
    public String toString() {
        return MoreObjects.toStringHelper(this)
                .add("i", i)
                .add("j", j)
                .toString();
    }
}
