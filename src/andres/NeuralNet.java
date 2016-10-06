package andres;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.security.SecureRandom;

/**
 * @author Andrés Funes<br/>
 *         Red Neuronal Artificial con algoritmo de aprendizaje Backpropagation
 *         con momento.
 */
public class NeuralNet
{
	private final class Layer
	{
		public double[] out;
		public double[] error;
		public double[][] weight;
		public double[][] deltaWeight;
	}

	private Layer[] layer;
	private Layer input;
	private Layer output;
	private double alpha;
	private double eta;

	/**
	 * Obtiene el porcentaje del deltaW a utilizar en el algoritmo
	 * backpropagation
	 * 
	 * @return double
	 */
	public double getAlpha()
	{
		return alpha;
	}

	/**
	 * Asigna el porcentaje del deltaW a utilizar en el algoritmo
	 * backpropagation
	 * 
	 * @param alpha
	 */
	public void setAlpha(double alpha)
	{
		this.alpha = alpha;
	}

	/**
	 * Obtiene el indice de aprendizaje a utilizar en el algoritmo
	 * backpropagation
	 * 
	 * @return double
	 */
	public double getEta()
	{
		return eta;
	}

	/**
	 * Asigna el indice de aprendizaje a utilizar en el algoritmo
	 * backpropagation
	 * 
	 * @param eta
	 */
	public void setEta(double eta)
	{
		this.eta = eta;
	}

	/**
	 * Crea una Red Neuronal
	 * 
	 * @param units
	 *            Array con la cantidad de unidades de cada capa de la red
	 *            neuronal
	 */
	public NeuralNet(int[] units)
	{
		init(units);
		alpha = 0.7;
		eta = 0.35;
	}

	/**
	 * Crea una Red Neuronal desde archivo .ann
	 * 
	 * @param filename
	 *            Archivo .ann que contiene los parametros y pesos de la red
	 *            neuronal
	 */
	public NeuralNet(String filename) throws IOException
	{
		load(filename);
	}

	private void init(int[] units)
	{
		if (units == null || units.length < 3)
			throw new NullPointerException("Invalid units argument. " + units);
		for (int i = 0; i < units.length; i++)
			if (units[i] < 1)
				throw new NullPointerException("Invalid units argument. " + units);
		SecureRandom rnd = new SecureRandom();
		layer = new Layer[units.length];
		for (int i = 0; i < units.length; i++)
		{
			layer[i] = new Layer();
			layer[i].out = new double[units[i] + 1];
			layer[i].error = new double[units[i] + 1];
			layer[i].out[0] = -1.0;
			if (i != 0)
			{
				layer[i].weight = new double[units[i] + 1][];
				layer[i].deltaWeight = new double[units[i] + 1][];
				for (int j = 1; j <= units[i]; j++)
				{
					layer[i].weight[j] = new double[units[i - 1] + 1];
					layer[i].deltaWeight[j] = new double[units[i - 1] + 1];
					for (int k = 0; k <= units[i - 1]; k++)
						layer[i].weight[j][k] = rnd.nextDouble() - 0.5;
				}
			}
		}
		input = layer[0];
		output = layer[layer.length - 1];
	}

	/**
	 * Obtiene la salida de la evaluación de la red neuronal con la entrada x[]
	 * 
	 * @param x
	 * @return double[]
	 */
	public double[] evaluate(double[] x)
	{
		for (int i = 1; i < input.out.length; i++)
			input.out[i] = x[i - 1];
		for (int i = 0; i < layer.length - 1; i++)
			for (int j = 1; j < layer[i + 1].out.length; j++)
			{
				double sum = 0;
				for (int k = 0; k < layer[i].out.length; k++)
					sum += layer[i].out[k] * layer[i + 1].weight[j][k];
				layer[i + 1].out[j] = 1.0 / (1.0 + Math.exp(-sum));
			}
		double[] y = new double[output.out.length - 1];
		for (int i = 1; i < output.out.length; i++)
			y[i - 1] = output.out[i];
		return y;
	}

	/**
	 * Evalua la red neuronal para una entrada x[] y calcula el Error de la
	 * misma al comparar con la salida esperada y[]
	 * 
	 * @param x
	 * @param y
	 * @return double
	 */
	public double getError(double[] x, double[] y)
	{
		double[] o = evaluate(x);
		double error = 0.0;
		for (int i = 1; i < output.out.length; i++)
		{
			double e = y[i - 1] - o[i - 1];
			error += 0.5 * e * e;
		}
		return error;
	}

	/**
	 * Evalua la red neuronal y corrige los pesos de la misma
	 * 
	 * @param x
	 * @param y
	 */
	public void train(double[] x, double[] y)
	{
		double[] o = evaluate(x);
		for (int i = 1; i < output.out.length; i++)
			output.error[i] = o[i - 1] * (1.0 - o[i - 1]) * (y[i - 1] - o[i - 1]);
		for (int i = layer.length - 1; i > 1; i--)
			for (int j = 1; j < layer[i - 1].out.length; j++)
			{
				double e = 0;
				for (int k = 1; k < layer[i].out.length; k++)
					e += layer[i].weight[k][j] * layer[i].error[k];
				layer[i - 1].error[j] = layer[i - 1].out[j] * (1 - layer[i - 1].out[j]) * e;
			}
		for (int i = 1; i < layer.length; i++)
			for (int j = 1; j < layer[i].out.length; j++)
				for (int k = 0; k < layer[i - 1].out.length; k++)
				{
					double dw = layer[i].deltaWeight[j][k];
					layer[i].deltaWeight[j][k] = eta * layer[i].error[j] * layer[i - 1].out[k];
					layer[i].weight[j][k] += layer[i].deltaWeight[j][k] + alpha * dw;
				}
	}

	/**
	 * Guarda la red neuronal en un archivo en formato ANN
	 * 
	 * @param filename
	 * @throws IOException
	 */
	public void save(String filename) throws IOException
	{
		FileOutputStream fos = new FileOutputStream(filename);
		DataOutputStream dos = new DataOutputStream(fos);
		byte[] b = new byte[3];
		b[0] = 'A';
		b[1] = 'N';
		b[2] = 'N';
		dos.write(b, 0, 3);
		dos.writeDouble(eta);
		dos.writeDouble(alpha);
		dos.writeInt(layer.length);
		for (int i = 0; i < layer.length; i++)
			dos.writeInt(layer[i].out.length - 1);
		for (int i = 1; i < layer.length; i++)
			for (int j = 1; j < layer[i].out.length; j++)
				for (int k = 0; k < layer[i - 1].out.length; k++)
					dos.writeDouble(layer[i].weight[j][k]);
		dos.close();
	}

	private void load(String filename) throws IOException
	{
		FileInputStream fis = new FileInputStream(filename);
		DataInputStream dis = new DataInputStream(fis);
		byte[] b = new byte[3];
		dis.read(b, 0, 3);
		if (b[0] != 'A' || b[1] != 'N' || b[2] != 'N')
		{
			dis.close();
			throw new IOException("Formato de archivo incorrecto.");
		}
		eta = dis.readDouble();
		alpha = dis.readDouble();
		int[] units = new int[dis.readInt()];
		for (int i = 0; i < units.length; i++)
			units[i] = dis.readInt();
		init(units);
		for (int i = 1; i < layer.length; i++)
			for (int j = 1; j < layer[i].out.length; j++)
				for (int k = 0; k < layer[i - 1].out.length; k++)
					layer[i].weight[j][k] = dis.readDouble();
		dis.close();
	}
}
