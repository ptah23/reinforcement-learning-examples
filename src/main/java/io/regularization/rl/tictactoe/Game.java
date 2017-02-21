package io.regularization.rl.tictactoe;

import com.sun.tools.doclint.Env;

import java.util.Map;
import java.util.Scanner;

/**
 * Created by ptah on 16/02/2017.
 */
public class Game {

    public void playGame(Agent p1, Agent p2, Environment environment, boolean draw) {
        // loops until the game is over
        Agent currentPlayer = null;
        while (!environment.gameOver(false)) {
            // alternate between players
            // p1 always starts first
            if (currentPlayer == p1) {
                currentPlayer = p2;
            } else {
                currentPlayer = p1;
            }



            // current player makes a move
            currentPlayer.takeAction(environment);

            // update state histories
            int state = environment.getState();
            p1.updateStateHistory(state);
            p2.updateStateHistory(state);
            // draw the board before the user who wants to see it makes a move
            if (draw) {

                environment.drawBoard();


            }

        }
        System.out.println("winner:" + environment.getWinner());
            // do the value function update
            p1.update(environment);
            p2.update(environment);

    }

    public static void main(String args[]) {
        //train the agent
        ComputerAgent p1 = new ComputerAgent();
        ComputerAgent p2 = new ComputerAgent();

        //set initial V for p1 and p2
        Environment environment = new Environment();
        Map<Integer, Integer> stateWinner = environment.getStateHashAndWinner(0, 0);
        p1.setSymbol(Environment.x);
        p2.setSymbol(Environment.o);

        p1.initialiseV(stateWinner);
        p2.initialiseV(stateWinner);
        int T = 10000;
        for (int t = 0; t < T; t++) {
            if (t % 200 == 0) {
                System.out.println(t);
            }
            Game game = new Game();
            game.playGame(p1, p2, new Environment(), false);

        }
        //# play human vs. agent
        //# do you think the agent learned to play the game well?
        HumanAgent human = new HumanAgent();
        human.setSymbol(Environment.o);
        while (true) {
            // p1.set_verbose(True)
            Game humanGame = new Game();
            humanGame.playGame(p1, human, new Environment(), true);
            // I made the agent player 1 because I wanted to see if it would
            // select the center as its starting move. If you want the agent
            //to go second you can switch the human and AI.
            System.out.println("Play again? [Y/n]: ");
            Scanner scanner = new Scanner(System.in);
            String answer = scanner.nextLine();

            if (answer.toLowerCase().contains("n")) {
                break;
            }

        }
    }
}
