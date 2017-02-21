package io.regularization.rl.tictactoe;

import java.util.*;

/**
 * Created by ptah on 16/02/2017.
 */
public class ComputerAgent implements Agent {
    private Random random = new Random();
    private double epsilon = 0.1;
    private double alpha = 0.5;
    private boolean verbose = false;
    private List<Integer> stateHistory= new ArrayList<>();
    private Map<Integer, Double> V = new HashMap<>();
    private int symbol;

    @Override
    public void resetHistory() {
        stateHistory.clear();
    }

    public void initialiseV(Map<Integer, Integer> stateWinner) {
        //initialize state values as follows
        //if x wins, V(s) = 1
        // if x loses or draw, V(s) = 0
        // otherwise, V(s) = 0.5
        stateWinner.entrySet().forEach(integerIntegerEntry ->  {
            if (integerIntegerEntry.getValue() == symbol) {
                V.put(integerIntegerEntry.getKey(), 1.0);
            } else if (integerIntegerEntry.getValue() == -symbol) {
                V.put(integerIntegerEntry.getKey(), 0.0);
            } else {
                V.put(integerIntegerEntry.getKey(), 0.5);
            }
        });
    }
    @Override
    public void takeAction(Environment environment) {

        double r = random.nextDouble();
        Integer[] nextMove = null;
        List<Integer[]> possibleMoves = environment.possibleMoves();
        if(r < epsilon) {

            int index = random.nextInt(possibleMoves.size() );
            if(verbose) {
                System.out.println("Taking a epsilon action:possibleMoves[" + index + "] from possibleMoves of size:" + possibleMoves.size());
            }
            nextMove = possibleMoves.get(index);
            if(verbose) {
                System.out.println("nextMove=" + nextMove);
            }
        } else {
            double bestValue = -1;
            for (Integer[] position : possibleMoves) {
                if(verbose) {
                    System.out.println("Taking a greedy action checking:position[" + position[0] + "][ " + position[1] + "]from possibleMoves of size:" + possibleMoves.size());
                }
                environment.makeMove(symbol, position);
                int state = environment.getState();
                environment.makeMove(0, position);
                if (V.get(state) > bestValue) {
                    bestValue = V.get(state);
                    nextMove = position;
                    if(verbose) {
                        System.out.println("Taking a greedy action choosing:position[" + position[0] + "][ " + position[1] + "] as nextMove");
                    }
                }
            }
            if(nextMove == null) {
                nextMove = possibleMoves.get(0);
            }
        }
        environment.makeMove(symbol, nextMove);
    }

    @Override
    public void updateStateHistory(int state) {
        stateHistory.add(state);
    }

    @Override
    public void update(Environment environment) {
        int reward = environment.reward(symbol);
        double target = reward;
        for(int i = stateHistory.size()-1; i >=0; i--) {
            int previous = stateHistory.get(i);
            double value = V.get(previous) + alpha * (target - V.get(previous));
            V.put(previous, value);
            target = value;
        }
        resetHistory();
    }

    public void setRandom(Random random) {
        this.random = random;
    }
    public void setSymbol(int symbol) {
        this.symbol = symbol;
    }

}
