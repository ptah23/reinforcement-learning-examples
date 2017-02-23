package io.regularization.rl.environment;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by ptah on 22/02/2017.
 */
public class GridWorldActionTest {

    @Test
    public void performUp() throws Exception {
        GridWorldPosition position = new GridWorldPosition(1,0);
        position = GridWorldAction.UP.perform(position);
        assertEquals(0, position.getI());
        assertEquals(0, position.getJ());
    }

    @Test
    public void undoUp() throws Exception {
        GridWorldPosition position = new GridWorldPosition(1,0);
        position = GridWorldAction.UP.undo(position);
        assertEquals(2, position.getI());
        assertEquals(0, position.getJ());
    }

    @Test
    public void performDown() throws Exception {
        GridWorldPosition position = new GridWorldPosition(1,0);
        position = GridWorldAction.DOWN.perform(position);
        assertEquals(2, position.getI());
        assertEquals(0, position.getJ());
    }

    @Test
    public void undoDown() throws Exception {
        GridWorldPosition position = new GridWorldPosition(2,0);
        position = GridWorldAction.DOWN.undo(position);
        assertEquals(1, position.getI());
        assertEquals(0, position.getJ());

    }
    @Test
    public void performLeft() throws Exception {
        GridWorldPosition position = new GridWorldPosition(0,1);
        position = GridWorldAction.LEFT.perform(position);
        assertEquals(0, position.getI());
        assertEquals(0, position.getJ());

    }
    @Test
    public void undoLeft() throws Exception {
        GridWorldPosition position = new GridWorldPosition(0,1);
        position = GridWorldAction.LEFT.undo(position);
        assertEquals(0, position.getI());
        assertEquals(2, position.getJ());
    }

    @Test
    public void performRight() throws Exception {
        GridWorldPosition position = new GridWorldPosition(0,1);
        position = GridWorldAction.RIGHT.perform(position);
        assertEquals(0, position.getI());
        assertEquals(2, position.getJ());
    }

    @Test
    public void undoRight() throws Exception {
        GridWorldPosition position = new GridWorldPosition(0,2);
        position = GridWorldAction.RIGHT.undo(position);
        assertEquals(0, position.getI());
        assertEquals(1, position.getJ());

    }


}