package io.regularization.rl.environment;

import com.google.common.base.MoreObjects;

/**
 * Created by ptah on 22/02/2017.
 */
public class GridWorldReward {
    private float value;
    public GridWorldReward(float value) {
        this.value = value;
    }

    public float getValue() {
        return value;
    }

    @Override
    public String toString() {
        return MoreObjects.toStringHelper(this)
                .add("value", value)
                .toString();
    }
}
