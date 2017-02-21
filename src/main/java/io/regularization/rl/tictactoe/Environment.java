package io.regularization.rl.tictactoe;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.EqualsCondition;
import org.nd4j.linalg.indexing.conditions.NotEqualsCondition;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by ptah on 16/02/2017.
 */
public class Environment {
    public static short LENGTH = 3;
    public static int NUMBER_OF_STATES = 3^(LENGTH*LENGTH);
    public static int x = -1;
    public static int o = 1;
    private int winner;
    private boolean ended;
    private INDArray board = Nd4j.zeros(LENGTH, LENGTH);


    public List<Integer[]> possibleMoves() {
        List<Integer[]> returnValue = new ArrayList<>();
        for (int i = 0; i < LENGTH; i++) {
            for (int j = 0; j < LENGTH; j++) {
                if (board.getInt(i, j) == 0) {
                    Integer[] emptySpace = {i, j};
                    returnValue.add(emptySpace);
                }
            }
        }
        return returnValue;
    }

    public void makeMove(int symbol, Integer[] nextMove) {
        board.putScalar(nextMove[0], nextMove[1], symbol);
    }


    public int getState() {
        int k = 0;
        int h = 0;
        for (int i = 0; i < LENGTH; i++) {
            for (int j = 0; j < LENGTH; j++) {
                int v = 0;
                if (board.getInt(i, j) == x) {
                    v = 1;
                } else if (board.getInt(i, j) == o) {
                    v = 2;
                }
                h += (3 ^ k) * v;
                k++;
            }
        }
        return h;
    }

    public int reward(int symbol) {

        if (!gameOver(false)) {
            return 0;
        } else {
            return winner == symbol ? 1 : 0;
        }

    }

    public boolean gameOver(boolean forceRecalculate) {
        if (!forceRecalculate && ended) {
            return ended;
        } else {
            ended = false;
        }
        INDArray sumOfColumns = board.sum(0);
        //columns
        if (checkForWinner(sumOfColumns, x)) {
            return ended;
        }
        if (checkForWinner(sumOfColumns, o)) {
            return ended;
        }
        //rows
        INDArray sumOfRows = board.sum(1);
        if (checkForWinner(sumOfRows, x)) {
            return ended;
        } else if (checkForWinner(sumOfRows, o)) {
            return ended;
        }

        //diagonals
        int diagonalSum = Nd4j.diag(board).sumNumber().intValue();
        if (checkDiagonal(diagonalSum, x)) {
            return ended;
        }
        if (checkDiagonal(diagonalSum, o)) {
            return ended;
        }
        int secondDiagonalSum = Nd4j.diag(Nd4j.rot(board)).sumNumber().intValue();
        if (checkDiagonal(secondDiagonalSum, x)) {
            return ended;
        }
        if (checkDiagonal(secondDiagonalSum, o)) {
            return ended;
        }
        //draw?
        if ((((int) board.cond(new EqualsCondition(0)).sumNumber().intValue()) == 0)) {
            winner = 0;
            ended = true;
        }
        return ended;

    }

    private boolean checkDiagonal(int diagonalSum, int symbol) {
        if (diagonalSum == LENGTH * symbol) {
            winner = symbol;
            ended = true;
        }
        return ended;
    }

    private boolean checkForWinner(INDArray sums, int player) {
        if (sums.getInt(0) == LENGTH * player ||
                sums.getInt(1) == LENGTH * player ||
                sums.getInt(2) == LENGTH * player) {
            winner = player;
            ended = true;
        }
        return ended;
    }

    public void drawBoard() {
        System.out.println("-------------");
        for(int i = 0; i < LENGTH; i++) {

            for (int j = 0; j < LENGTH; j++) {
                System.out.print(" ");
                if (board.getInt(i, j) == x) {
                    System.out.print("x");
                } else if (board.getInt(i, j) == o) {
                    System.out.print("o");
                } else {
                    System.out.print(" ");
                }
            }
            System.out.print("\n");
        }
        System.out.println("-------------");
    }

    public Map<Integer, Integer> getStateHashAndWinner(int i, int j) {
        Map<Integer, Integer> results = new HashMap<>();
        List<Integer> states = new ArrayList<>();
        states.add(0);
        states.add(x);
        states.add(o);
        for (int v :states) {
            makeMove(v, new Integer[]{i, j}); //#if empty board it should already be 0
            if (j == 2) {
                //j goes back to 0, increase i, unless i = 2, then we are done
                if (i == 2) {
                    //the board is full, collect results and return
                    int state = getState();
                    gameOver(true);

                    if (ended) {
                        results.put(state, winner);
                        //drawBoard();
                    } else {
                        results.put(state, 0);
                    }
                } else {
                    results.putAll(getStateHashAndWinner(i + 1, 0));
                }
            } else {
                //increment j, i stays the same
                results.putAll(getStateHashAndWinner(i, j + 1));
            }
        }
        return results;
    }

    public INDArray getBoard() {
        return board;
    }

    public void setBoard(INDArray board) {
        this.board = board;
    }

    public boolean isEmpty(int i, int j) {
        return board.getInt(i,j) == 0;
    }

    public int getWinner() {
        return winner;
    }
}