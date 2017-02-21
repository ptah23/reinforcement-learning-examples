package io.regularization.rl.tictactoe;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.MathUtils;


import static org.junit.Assert.*;

/**
 * Created by ptah on 20/02/2017.
 */
public class EnvironmentTest {

    @Test
    public void makeMoveOneOne() throws Exception {
        Integer[] position = {0, 0};
        checkMakeMoveBothSymbols(position);
    }

    @Test
    public void makeMoveOneTwo() throws Exception {
        Integer[] position = {0, 1};
        checkMakeMoveBothSymbols(position);
    }

    @Test
    public void makeMoveOneThree() throws Exception {
        Integer[] position = {0, 2};
        checkMakeMoveBothSymbols(position);
    }

    @Test
    public void makeMoveTwoOne() throws Exception {
        Integer[] position = {1, 0};
        checkMakeMoveBothSymbols(position);
    }

    @Test
    public void makeMoveTwoTwo() throws Exception {
        Integer[] position = {1, 1};
        checkMakeMoveBothSymbols(position);
    }

    @Test
    public void makeMoveTwoThree() throws Exception {
        Integer[] position = {1, 2};
        checkMakeMoveBothSymbols(position);
    }

    @Test
    public void makeMoveThreeOne() throws Exception {
        Integer[] position = {2, 0};
        checkMakeMoveBothSymbols(position);

    }
    @Test
    public void makeMoveThreeTwo() throws Exception {
        Integer[] position = {2, 1};
        checkMakeMoveBothSymbols(position);
    }

    @Test
    public void makeMoveThreeThree() throws Exception {
        Integer[] position = {2, 2};
        checkMakeMoveBothSymbols(position);
    }

    private void assertZeroExcept(INDArray board, Integer[] position) {
        for(int i = 0; i< Environment.LENGTH;i++) {
            for(int j = 0; j< Environment.LENGTH;j++) {
                Integer[] currentPosition = {i, j};
                if(currentPosition[0] != position[0] || currentPosition[1] != position[1]) {
                    assertEquals("currentPosition:" + currentPosition[0] + ", " + currentPosition[1],0, board.getInt(currentPosition[0], currentPosition[1]));
                }
            }
        }
    }

    private void checkMakeMove(int symbol, Integer[] position) {
        Environment environment = new Environment();
        environment.makeMove(symbol, position);
        INDArray board = environment.getBoard();
        assertEquals(symbol, board.getInt(position[0],position[1]));
        assertZeroExcept(board, position);
    }
    private void checkMakeMoveBothSymbols(Integer[] position) {
        checkMakeMove(Environment.x, position);
        checkMakeMove(Environment.o, position);
    }

    @Test
    public void getStateConsistent() throws Exception {
        Environment environment = new Environment();
        Integer[] position = {1, 1};
        environment.makeMove(Environment.x, position);
        int stateOne =environment.getState();
        Environment environmentTwo = new Environment();
        Integer[] positionTwo = {1, 1};
        environmentTwo.makeMove(Environment.x, positionTwo);
        int stateTwo = environmentTwo.getState();
        assertEquals(stateOne,stateTwo);
    }

    @Test
    public void reward() throws Exception {
        checkRowWin(0, Environment.x, Environment.o);
        checkRowWin(0, Environment.o, Environment.x);
        checkRowWin(1, Environment.x, Environment.o);
        checkRowWin(1, Environment.o, Environment.x);
        checkRowWin(2, Environment.x, Environment.o);
        checkRowWin(2, Environment.o, Environment.x);

        checkColumnWin(0, Environment.x, Environment.o);
        checkColumnWin(0, Environment.o, Environment.x);
        checkColumnWin(1, Environment.x, Environment.o);
        checkColumnWin(1, Environment.o, Environment.x);
        checkColumnWin(2, Environment.x, Environment.o);
        checkColumnWin(2, Environment.o, Environment.x);

        checkLeftDiagonalWin(Environment.x, Environment.o);
        checkLeftDiagonalWin(Environment.o, Environment.x);
        checkRightDiagonalWin(Environment.x, Environment.o);
        checkRightDiagonalWin(Environment.o, Environment.x);
    }

    private void checkLeftDiagonalWin(int winner, int loser) {
        Environment environment = new Environment();
        Integer[] position = {0, 0};
        environment.makeMove(winner, position);
        position = new Integer[]{0, 1};
        environment.makeMove(loser, position);
        position = new Integer[]{1, 1};
        environment.makeMove(winner, position);
        position = new Integer[]{0, 2};
        environment.makeMove(loser, position);
        position = new Integer[]{2, 2};
        environment.makeMove(winner, position);
        assertEquals(1, environment.reward(winner));
        assertEquals(0, environment.reward(loser));
    }
    private void checkRightDiagonalWin(int winner, int loser) {
        Environment environment = new Environment();
        Integer[] position = {2, 0};
        environment.makeMove(winner, position);
        position = new Integer[]{1, 1};
        environment.makeMove(winner, position);
        position = new Integer[]{0, 2};
        environment.makeMove(winner, position);
        assertEquals(1, environment.reward(winner));
        assertEquals(0, environment.reward(loser));
    }

    private void checkRowWin(int rowNumber, int winner, int loser) {
        Environment environment = new Environment();
        Integer[] position = {rowNumber, 0};
        environment.makeMove(winner, position);
        position = new Integer[]{rowNumber, 1};
        environment.makeMove(winner, position);
        position = new Integer[]{rowNumber, 2};
        environment.makeMove(winner, position);
        assertEquals(1, environment.reward(winner));
        assertEquals(0, environment.reward(loser));
    }

    private void checkColumnWin(int columnNumber, int winner, int loser) {
        Environment environment = new Environment();
        Integer[] position = {0, columnNumber};
        environment.makeMove(winner, position);
        position = new Integer[]{1, columnNumber};
        environment.makeMove(winner, position);
        position = new Integer[]{2, columnNumber};
        environment.makeMove(winner, position);
        assertEquals(1, environment.reward(winner));
        assertEquals(0, environment.reward(loser));
    }

    @Test
    public void getStateHashAndWinner() {
        Environment environment = new Environment();
        environment.getStateHashAndWinner(0,0);
    }
    @Test
    public void gameOver() throws Exception {

    }

    @Test
    public void possibleMovesOneSpaceLeft() throws Exception {
        Environment environment = new Environment();
        environment.makeMove(Environment.x, new Integer[]{0,0});
        environment.makeMove(Environment.x, new Integer[]{0,1});
        environment.makeMove(Environment.x, new Integer[]{0,2});
        environment.makeMove(Environment.x, new Integer[]{1,0});
        environment.makeMove(Environment.x, new Integer[]{1,1});
        environment.makeMove(Environment.x, new Integer[]{1,2});
        environment.makeMove(Environment.x, new Integer[]{2,0});
        environment.makeMove(Environment.x, new Integer[]{2,1});
        assertEquals(1, environment.possibleMoves().size());
    }

}