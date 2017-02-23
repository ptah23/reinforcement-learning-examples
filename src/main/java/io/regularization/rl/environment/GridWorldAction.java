package io.regularization.rl.environment;

/**
 * Created by ptah on 22/02/2017.
 */
public enum GridWorldAction {
    UP(-1, 0),
    DOWN(1, 0),
    RIGHT(0, 1),
    LEFT(0, -1);
    private int incrementI = 0;
    private int incrementJ = 0;

    GridWorldAction(int incrementI, int incrementJ) {
        this.incrementI = incrementI;
        this.incrementJ = incrementJ;
    }

    public GridWorldPosition perform(GridWorldPosition currentPosition) {
            return new GridWorldPosition(currentPosition.getI() + incrementI,
                    currentPosition.getJ() + incrementJ);
    }

    public GridWorldPosition undo(GridWorldPosition currentPosition) {
        return new GridWorldPosition(currentPosition.getI() - incrementI,
                currentPosition.getJ() - incrementJ);
    }
}
