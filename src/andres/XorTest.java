package andres;

import java.security.SecureRandom;

public class XorTest
{
	public static void main(String[] args)
	{
		int[] units = { 2, 3, 1};
		NeuralNet nn = new NeuralNet(units);
		double[][][] cases = {{{ 0, 0}, {0}}, {{ 0, 1}, {1}}, {{ 1, 0}, {1}}, {{ 1, 1}, {0}}};
		SecureRandom rnd = new SecureRandom();
		int n, i;
		double error = 1;
		for (i = 0; error > 0.0001 && i < 100000000; i++)
		{
			n = rnd.nextInt(4);
			nn.train(cases[n][0], cases[n][1]);
			error = getError(nn, cases);
		}
		System.out.println("Iterations = " + i);
		for (i = 0; i < 4; i++)
			System.out.println(cases[i][0][0] + " " + cases[i][0][1] + " = " + nn.evaluate(cases[i][0])[0] + ", error = " + nn.getError(cases[i][0], cases[i][1]));
	}
	
	private static double getError(NeuralNet nn, double[][][] cases)
	{
		double error = 0.0d;
		for (int i = 0; i < 4; i++)
			error += nn.getError(cases[i][0], cases[i][1]);
		return error / 4;
	}
}
