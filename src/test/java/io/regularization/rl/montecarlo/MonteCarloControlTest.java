package io.regularization.rl.montecarlo;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import io.regularization.rl.environment.*;
import org.junit.Test;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static io.regularization.rl.environment.GridWorldAction.RIGHT;
import static io.regularization.rl.environment.GridWorldAction.UP;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/**
 * Created by ptah on 14/03/2017.
 */
public class MonteCarloControlTest {

    @Test
    public void playRound() {
        GridWorldEnvironment grid = GridWorldEnvironment.negativeGrid();
        Map<GridWorldState, GridWorldAction> policy = ImmutableMap.<GridWorldState, GridWorldAction>builder()
                .put(new GridWorldState(2, 0), UP)
                .put(new GridWorldState(1, 0), UP)
                .put(new GridWorldState(0, 0), RIGHT)
                .put(new GridWorldState(0, 1), RIGHT)
                .put(new GridWorldState(0, 2), RIGHT)
                .put(new GridWorldState(1, 2), RIGHT)
                .put(new GridWorldState(2, 1), RIGHT)
                .put(new GridWorldState(2, 2), RIGHT)
                .put(new GridWorldState(2, 3), UP).build();
        GridWorldState start = new GridWorldState(1, 2);
        grid.setCurrentPosition(start);
        LinkedHashMap<GridWorldStateAction, GridWorldReward> statesAndRewards = Maps.newLinkedHashMap();
        statesAndRewards.put(new GridWorldStateAction(start, policy.get(start)), grid.getRewards().get(start));
        GridWorldAction action = ControlExploringStarts.playRound(grid, policy, new Random(), statesAndRewards);
        assertNull(action);
        assertEquals(2, statesAndRewards.size());
        assertEquals(-1.0f, statesAndRewards.get(new GridWorldStateAction(new GridWorldState(1, 3), null)).getValue(), 0.001f);
    }

    @Test
    public void calculateG() {
        //ControlExploringStarts.calculateG()
    }

}