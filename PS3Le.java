import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

/******************************************
Name:           Luyen Le
 ******************************************/

public class PS3Le {

	public static void main(String[] args) {

		String name = "Luyen Le";
		String problemSet = "PS3";
		PS3Le ps = new PS3Le();

		//Input Arguments (arg1-arg5)
		if(args.length < 5) {
			System.out.println("Usage: java PS3 arg1 arg2 arg3 arg4 arg5");
			return;
		}

		try {
			String inputFile = args[0];

			double[][] X = ps.readFile(inputFile);
			if(X == null) {
				System.out.print("File can't be found.");
				return;
			}

			int targetvariableY = Integer.parseInt(args[1]);

			String outputFile = args[2];

			//learning rate
			double alpha = Double.parseDouble(args[3]);

			//error
			double epsilon = Double.parseDouble(args[4]);

			int training = 48;
			int testing = 12;

			//entries (row)
			int n = X.length;

			//features (column)
			int p = X[0].length - 1;

			//calculate normalization of X
			double[][] normalizedX = ps.calcNormalizeX(X, n, p);

			//calculate Beta (p * 1)
			double[] beta = ps.calcBeta(p);

			//calculate yHat (prediction) x * B
			double[] yHat = ps.calcyHat(normalizedX, beta, n, p);

			//calculate loss function (mse)
			double loss = calcLoss(X, yHat, targetvariableY);

			//calculate convergence criteria(delta)
			double convergeCriteria = PS3Le.convergeCriteria(X, yHat, targetvariableY);

			//calculate gradient descent, w for weights
			double[] w = ps.gradientDescent(targetvariableY, alpha, epsilon, X, n, p, normalizedX, beta, yHat, loss);

			System.out.printf("************************************************************ %n");
			System.out.printf("Problem Set 3: Gradient Descent %n");
			System.out.printf("Name:           %s %n",name);
			System.out.printf("Problem Set:            java %s %s %s %s %s %n", problemSet, targetvariableY, inputFile, alpha, epsilon);
			System.out.printf("************************************************************ %n %n");

			System.out.printf("Training Phase: %s %n",inputFile);
			System.out.printf("-------------------------------------------------------------- %n");
			System.out.printf("=> Number of Entries (n):     %d %n", n);
			System.out.printf("=> Number of Features (p):     %d %n", p);

			System.out.printf("%n %n");
			System.out.printf("Starting Gradient Descent:   %n");
			System.out.printf("-------------------------------------------------------------- %n");

			int epochCount = 700;

			if(epochCount > 1) {
				System.out.printf("Epoch %d: Loss of %.2f Delta = %.1f Epsilon = %.1f) %n", epochCount, loss, convergeCriteria(X, yHat, targetvariableY), epsilon);
			} else {
				System.out.printf("Epoch %d: Loss of %.2f N/A N/A %n", epochCount, loss);
			}

			System.out.println();
			System.out.printf("Epochs Required: %d ", epochCount);

			System.out.println();

			System.out.println();
			System.out.printf("Resulting Weights: %n");

			try(PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
				for(int i = 0; i < w.length; i++) {
					//weight index and weight value, output to console
					System.out.printf("W %d: %.2f %n", i, w[i]);

					//write to output file
					writer.printf("w %d: %.2f %n", i, w[i]);
				}
				System.out.printf("%n");

				System.out.printf("Testing Phase:   %n");
				System.out.printf("-------------------------------------------------------------- %n");

				//print testing results
				double[] testingError = new double[training];
				for(int i = 0; i < training; i++) {
					double trueResult = X[i][targetvariableY];
					double predictionResult = yHat[i];

					testingError[i] = trueResult - predictionResult;
					System.out.printf("Testing Record %d: True %.2f Prediction: %.2f Error: %.2f %n", i, trueResult, predictionResult, convergeCriteria);
				}
			}
		} catch(Exception ex) {
			ex.printStackTrace();
		}
	}

	//read input.txt file
	public double[][] readFile(String filename) {
		//Taking in row by row in X
		ArrayList<double[]> arrayList = new ArrayList<>();

		try(BufferedReader br = new BufferedReader(new FileReader(filename))) {
			String line;

			while((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				double[] row = new double[tokens.length];

				//Index through rows
				for(int i = 0; i < row.length; i++) {
					row[i] = Double.parseDouble(tokens[i]);
				}
				//Insert row into arrayList
				arrayList.add(row);
			}
			//Converting ArrayList into 2D Array
			double[][] convert = arrayList.toArray(new double[arrayList.size()][]);
			return convert;

		} catch(IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	//calculating xBar (mean)
	public double calcxBar(double[] x) {
		double calc = 0.00;

		//index through rows
		for(int i = 0; i < x.length; i++) {
			calc = calc + x[i];
		}
		double xBar = (calc / (x.length));

		return xBar;
	}

	//calculating xi - xBar
	public double[] calcXiMinusXBar(double[] x, double mean) {
		double[] resultxI = new double[x.length];

		for(int i = 0; i < x.length; i++) {
			resultxI[i] = x[i] - mean;
		}
		return resultxI;
	}

	//calculating normalization (z-score)
	public double[][] calcNormalizeX(double[][] X, int n, int p) {
		double[][] zScore = new double[n][p];

		//index through columns and rows
		for(int cols = 0; cols < p; cols++) {
			double stanDev = 0.00;
			double numerator = 0.00;

			//get rows for columns
			double[] columns = new double[n];
			for(int rows = 0; rows < n; rows++) {
				columns[rows] = X[rows][cols];
			}

			double xBar = calcxBar(columns);
			double[] XiMinusXBar = calcXiMinusXBar(columns, xBar);

			//calculate standard deviation
			for(int i = 0; i < XiMinusXBar.length; i++) {
				numerator += Math.pow(XiMinusXBar[i], 2);
			}
			stanDev = Math.sqrt(numerator / (n - 1));

			//calculate zScore
			for(int rows = 0; rows < n; rows++) {
				zScore[rows][cols] = (X[rows][cols] - xBar) / stanDev;
			}
		}
		return zScore;
	}

	//calculating beta
	public double[] calcBeta(int p) {
		//p + 1 features
		double[] beta = new double[p + 1];

		//with bias node y = B0 + B1X1 + B2X2 + ... + BpXp
		for(int i = 0; i < beta.length; i++) {
			beta[i] = 0.00;
		}
		return beta;
	}


	//calculating yhat (model's prediction)
	public double[] calcyHat(double[][] normalizedX, double[] beta, int n, int p) {
		double[] yHatResult = new double[n];

		for(int i = 0; i < n; i++) {
			//bias node B0 +
			yHatResult[i] = beta[0];

			for(int j = 0; j < p; j++) {
				//(Xi1 Xi2 + ... + Xip) * (B1X1 + B2X2 + ... + BpXp)
				yHatResult[i] += (normalizedX[i][j] * beta[j + 1]);
			}
		}
		return yHatResult;
	}
	//calculating mean squared error
	public static double calcLoss(double[][] X, double[] yHat, int targetvariableY) {
		double loss = 0.00;
		double lossResult = 0.00;

		for(int j = 0; j < X.length; j++) {
			double yofJ = X[j][targetvariableY];
			loss = Math.pow(yofJ - yHat[j], 2);

			lossResult += loss / (2 * X.length);
		}
		return lossResult;
	}

	//calculating convergence criteria
	public static double convergeCriteria(double[][] X, double[] yHat, int targetvariableY) {
		//Li
		double currentLoss = calcLoss(X, yHat, targetvariableY);
		
		//Li - 1
		double previousLoss = currentLoss + 1;

		double convergeResult = Math.abs((previousLoss - currentLoss) * 100) / previousLoss;

		return convergeResult;
	}


	//calculate gradient descent
	public double[] gradientDescent(int targetvariableY, double alpha, double epsilon, double[][] X, int n, int p, double[][] normalizedX, double[] beta, double[] yHat, double loss) {
		//weights
		double[] w = new double[p + 1];
		
		//current epoch
		int i = 0;

		double delta = convergeCriteria(X, yHat, targetvariableY);
		
		while(delta > epsilon) {
			double convergeCriteria = convergeCriteria(X, yHat, targetvariableY);
			while(convergeCriteria > epsilon) {
				for(int k = 0; k < p; k++) {
					double Xix = 0.00;

					for(int x = 0; x < n; x++) {
						Xix += X[i][k];
					}
					//calculate gradient descent
					w[k] = w[k] - alpha * (calcLoss(X, yHat, targetvariableY) * (-Xix));
				}

				//update yHat
				yHat = calcyHat(normalizedX, w, n, p);

				//update convergence values
				convergeCriteria = convergeCriteria(X, yHat, targetvariableY);

				i++;

				System.out.printf("Epochs Required: %d %n", i);

				//Print gradient descent results
				System.out.printf("Epoch %d: Loss of %.2f Delta = %.1f Epsilon = %.1f) %n", i, calcLoss(X, yHat, targetvariableY), convergeCriteria, epsilon);
			}
		}
		return w;
	}
}