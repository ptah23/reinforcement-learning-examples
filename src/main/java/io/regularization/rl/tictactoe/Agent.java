package io.regularization.rl.tictactoe;

/**
 * Created by ptah on 21/02/2017.
 */
public interface Agent {
    void resetHistory();

    void takeAction(Environment environment);

    void updateStateHistory(int state);

    void update(Environment environment);
    void setSymbol(int symbol);
}
