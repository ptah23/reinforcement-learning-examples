package io.regularization.rl.tictactoe;

import java.util.Scanner;

/**
 * Created by ptah on 21/02/2017.
 */
public class HumanAgent implements Agent {
    private int symbol;

    @Override
    public void resetHistory() {

    }

    @Override
    public void takeAction(Environment environment) {
        while(true) {
      //break if we make a legal move
            System.out.println("Enter coordinates i,j for your next move (i,j=0..2):");
            Scanner scanner = new Scanner(System.in);
            String move = scanner.nextLine();
            try {
                String[] coordinates = move.split(",");
                int i = Integer.parseInt(coordinates[0]);
                int j = Integer.parseInt(coordinates[1]);
                if (environment.isEmpty(i, j)) {
                    environment.makeMove(symbol, new Integer[]{i, j});
                }
            } catch(Exception e) {
                e.printStackTrace();
            }
            break;
        }
    }

    @Override
    public void updateStateHistory(int state) {

    }

    @Override
    public void update(Environment environment) {

    }

    @Override
    public void setSymbol(int symbol) {
        this.symbol = symbol;
    }
}
