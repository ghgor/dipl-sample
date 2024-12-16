package com.dipl.sample;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;


import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class KalmanFilterExample {

    private XYSeries positionSeries;
    private XYSeries velocitySeries;
    private double delta_t; // Time step duration
    private RealMatrix X; // State vector: [position, velocity]
    private RealMatrix X_hat; // Predicted state vector
    private RealMatrix P; // Predicted process covariance matrix
    private RealMatrix Q; // Process noise covariance matrix
    private RealMatrix R; // Measurement noise covariance matrix
    private RealMatrix K; // Kalman gain
    private RealMatrix A; // State transition matrix
    private RealMatrix B; // Control input matrix
    private RealMatrix H; // Observation matrix
    private RealMatrix C; // Observation matrix for the output equation
    private RealMatrix Z;
    public int currentIndex; // Index to track the current acceleration value
    public double [][] measurements;

    public RealMatrix eulerX;
    public RealMatrix eulerY;
    public RealMatrix eulerZ;

    public RealMatrix newAcc;


    public KalmanFilterExample(double deltaPx, double deltaPy, double deltaPxv, double deltaPyv, double delta_x, double delta_y, double delta_xv, double delta_yv, double delta_t, double[][] H, double[][] C, RealMatrix accelerations, RealMatrix eulerX, RealMatrix eulerY, RealMatrix eulerZ) {
        this.delta_t = delta_t; // Time step duration
        if (accelerations.getColumn(0).length < 3 ) {
            throw new IllegalArgumentException("At least 3 acceleration values are required to calculate measurements.");
        }

        // Prepare an array to store measurements
        this.measurements = new double[accelerations.getColumn(0).length - 2][4];


        // Calculate accelerations with rotation matrices
        this.eulerX = eulerX;
        this.eulerY = eulerY;
        this.eulerZ = eulerZ;

        this.newAcc = getR().multiply(accelerations.transpose()).transpose();



        // Calculate measurements
        for (int i = 0; i < this.measurements.length; i++) {

            double currentAccelerationX = newAcc.getRow(i)[0];
            double currentAccelerationY = newAcc.getRow(i)[1];
            double nextAccelerationX = newAcc.getRow(i + 1)[0];
            double nextAccelerationY = newAcc.getRow(i + 1)[1];
            double nextNextAccelerationX = newAcc.getRow(i + 2)[0];
            double nextNextAccelerationY = newAcc.getRow(i + 2)[1];

            double velocityX = trapezoidalRuleForVelocity(currentAccelerationX, nextAccelerationX);
            double velocityY = trapezoidalRuleForVelocity(currentAccelerationY, nextAccelerationY);
            double positionX = trapezoidalRuleForPosition(velocityX, trapezoidalRuleForVelocity(nextAccelerationX, nextNextAccelerationX));
            double positionY = trapezoidalRuleForPosition(velocityY, trapezoidalRuleForVelocity(nextAccelerationY, nextNextAccelerationY));


            this.measurements[i][0] = positionX;
            this.measurements[i][1] = positionY;
            this.measurements[i][2] = velocityX;
            this.measurements[i][3] = velocityY;
        }
        this.Q = new Array2DRowRealMatrix(new double[][]{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}); // Process noise covariance matrix (set to zero)
        this.R = new Array2DRowRealMatrix(new double[][]{{delta_x * delta_x, 0, 0, 0}, {0, delta_y * delta_y, 0, 0}, {0, 0, delta_xv * delta_xv, 0}, {0, 0, 0, delta_yv * delta_yv}}); // Measurement noise covariance matrix
        this.P = new Array2DRowRealMatrix(new double[][]{{deltaPx * deltaPx, 0, 0, 0}, {0, deltaPy * deltaPy, 0, 0}, {0, 0, deltaPxv * deltaPxv, 0}, {0, 0, 0, deltaPyv * deltaPyv}}); // Initial process covariance matrix
        this.X = new Array2DRowRealMatrix(new double[][]{{this.measurements[0][0]}, {this.measurements[0][1]}, {this.measurements[0][2]}, {this.measurements[0][3]}}); // Initial state: [positionX, positionY, velocityX, velocityY]
        this.A = new Array2DRowRealMatrix(new double [][] {{1, 0, delta_t, 0}, {0, 1, 0, delta_t}, {0, 0, 1, 0}, {0, 0, 0, 1}});  // State transition matrix
        this.B = new Array2DRowRealMatrix(new double [][] {{0.5 * delta_t * delta_t, 0}, {0, 0.5 * delta_t * delta_t}, {delta_t, 0}, {0, delta_t}}); // Control input matrix
        this.H = new Array2DRowRealMatrix(H); // Observation matrix
        this.C = new Array2DRowRealMatrix(C); // Observation matrix for the output equation
        this.currentIndex = 0; // Start with the first acceleration value

        positionSeries = new XYSeries("Position");
        velocitySeries = new XYSeries("Velocity");
    }


    public void update() {

        // 1: Calculate the predicted state
        RealMatrix newAccWithoutZ = new Array2DRowRealMatrix(new double [][] {newAcc.getColumn(0), newAcc.getColumn(1)}).transpose();
        X_hat = A.multiply(X).add(B.multiply((newAccWithoutZ.getRowMatrix(currentIndex)).transpose()));


        // 2: Calculate the predicted process covariance matrix
        P = A.multiply(P).multiply(A.transpose()).add(Q);


        // 3: Calculate the Kalman gain
        RealMatrix H_transpose = H.transpose();
        RealMatrix S = H.multiply(P).multiply(H_transpose).add(R);
        LUDecomposition lu = new LUDecomposition(S);
        K = P.multiply(H_transpose).multiply(lu.getSolver().getInverse());

        // 4: Calculate the new observation
        // Move to the next acceleration value
        this.currentIndex++;
        if(this.currentIndex == (newAcc.getColumn(0).length - 2))
        {
            return;
        }
        RealMatrix Y = C.multiply(new Array2DRowRealMatrix(new double[][]{{this.measurements[currentIndex][0]},
                                         {this.measurements[currentIndex][1]}, {this.measurements[currentIndex][2]},
                                         {this.measurements[currentIndex][3]}}));


        // 5: Calculate the current state
        X = X_hat.add(K.multiply(Y.subtract(H.multiply(X_hat))));


        // 6: Update the process covariance matrix
        RealMatrix I = MatrixUtils.createRealIdentityMatrix(P.getRowDimension());
        P = I.subtract(K.multiply(H)).multiply(P);


        System.out.println(currentIndex * delta_t);
        positionSeries.add(getState()[0], getState()[1]);
        velocitySeries.add(getState()[2], getState()[3]);

    }


    private double trapezoidalRuleForVelocity(double acceleration, double nextAcceleration) {
        return ((acceleration + nextAcceleration) / 2.0) * delta_t;
    }

    private double trapezoidalRuleForPosition(double currentVelocity, double nextVelocity) {
        return ((currentVelocity + nextVelocity) / 2.0) * delta_t;
    }



    public double[] getState() {
        return X.getColumn(0);
    }


    public void printMeasurements(double [] accelerations)
    {
        double [][] temp = this.measurements;
        for(int i = 0; i < temp.length; ++i)
        {
            for(int j = 0; j < temp[i].length; ++j)
            {
                System.out.print(temp[i][j] + " ");
            }
            System.out.println();
        }
    }


    public RealMatrix getR() {

        RealMatrix R = MatrixUtils.createRealIdentityMatrix(3);
        for(int i = 0; i < eulerX.getColumn(0).length; ++i) {
            RealMatrix Rx = new Array2DRowRealMatrix(new double [][] { {1, 0, 0 }, { 0, Math.cos(eulerX.getRow(i)[0]), -Math.sin(eulerX.getRow(i)[0]) },
                    { 0, Math.sin(eulerX.getRow(i)[0]), Math.cos(eulerX.getRow(i)[0]) } });

            RealMatrix Ry = new Array2DRowRealMatrix(new double [][] { { Math.cos(eulerY.getRow(i)[0]), 0, Math.sin(eulerY.getRow(i)[0]) }, { 0, 1, 0 },
                { -Math.sin(eulerY.getRow(i)[0]), 0, Math.cos(eulerY.getRow(i)[0]) } });

            RealMatrix Rz = new Array2DRowRealMatrix(new double [][] { { Math.cos(eulerZ.getRow(i)[0]), -Math.sin(eulerZ.getRow(i)[0]), 0 }, { Math.sin(eulerZ.getRow(i)[0]), Math.cos(eulerZ.getRow(i)[0]), 0 },
                    { 0, 0, 1 } });

           R  = Rz.multiply(Ry).multiply(Rx);


        }
        return R;
    }

    public void plotResults() {

        // Create a dataset for position and velocity series
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(positionSeries);
        //dataset.addSeries(velocitySeries);

        // Create the chart
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Kalman Filter Results",
                "Px",
                "Py",
                dataset
        );

        // Invert Y-axis
        // chart.getXYPlot().getRangeAxis().setInverted(true);

        chart.getXYPlot().getDomainAxis().setInverted(true);

        // Create and set up the frame
        JFrame frame = new JFrame("Kalman Filter Results");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout());

        // Add the chart to a panel
        ChartPanel chartPanel = new ChartPanel(chart);
        frame.add(chartPanel, BorderLayout.CENTER);

        // Display the frame
        frame.pack();
        frame.setVisible(true);

    }

    public static void main(String[] args) {
        // Initial values
        //double[] initialState = {4000, 280}; // Initial state: [position, velocity]
        double deltaPx = 20;
        double deltaPy = 20;
        double deltaPxv = 5;
        double deltaPyv = 5;

        double delta_x = 25;
        double delta_y = 25;
        double delta_xv = 6;
        double delta_yv = 6;
        double delta_t = 0.01;
        double[][] H = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}; // Observation matrix
        double[][] C = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}; // Observation matrix for the output equation
        // Example accelerations
        RealMatrix accelerations = new Array2DRowRealMatrix(new double[][]{  {0.000210158,0.001336694,-0.010737477},
                {0.002036421,-0.001830513,-0.012324274},
                {0.002149662,-0.000476537,-0.012568355},
                {0.000312362,0.001600924,-0.013300717},
                {-0.00042058,0.001352887,-0.013911067},
                {0.000312192,0.002567959,-0.013300894},
                {0.001773171,-0.000242778,-0.011347771},
                {0.000181439,-0.003040986,-0.012202203},
                {5.95E-05,-0.000102199,-0.010859309},
                {-0.002622006,0.000626158,-0.014155209},
                {-0.000905205,-0.000597155,-0.011591673},
                {0.001049784,0.000138294,-0.011347532},
                {0.004580246,-0.000955658,-0.012079952},
                {0.00273542,-0.001436798,-0.015742004},
                {-0.001048718,-0.000941652,-0.011713564},
                {-0.001893298,0.001379885,-0.009027896},
                {6.72E-05,-0.000938152,-0.010981023},
                {0.001404673,-0.003123334,-0.014276743},
                {-0.000429991,-0.001891128,-0.013422072},
                {-0.000671544,0.001647281,-0.009882091},
                {-0.001521006,0.002121617,-0.011957524},
                {0.000433711,0.000768279,-0.013178287},
                {0.000796994,-0.001796608,-0.012201786},
                {0.000185325,-0.00154787,-0.011103034},
                {0.00055239,0.001499576,-0.011957703},
                {6.34E-05,0.003681153,-0.014887571},
        });



        RealMatrix eulerX = new Array2DRowRealMatrix(new double [][] {
            {-0.866965592}, {-0.867344975}, {-0.867986798}, {-0.868116677},
            {-0.867893457}, {-0.867570758}, {-0.867391944}, {-0.867931008},
            {-0.868452489}, {-0.868219554}, {-0.868070126}, {-0.868243515},
            {-0.868511736}, {-0.868920863}, {-0.869314253}, {-0.869440496},
            {-0.869514883}, {-0.87020731}, {-0.870866477}, {-0.870773017},
            {-0.869973898}, {-0.869368553}, {-0.869287729}, {-0.869551241},
            {-0.869304061}, {-0.868405104},
        });

        RealMatrix eulerY = new Array2DRowRealMatrix(new double [][] {
            {0.284812808}, {0.284538209}, {0.284032375}, {0.283674508},
            {0.283644706}, {0.283664763}, {0.283443332}, {0.283167303},
            {0.283176512}, {0.283406287}, {0.283854246}, {0.283960968},
            {0.283412337}, {0.282623202}, {0.282625616}, {0.283193558},
            {0.283615321}, {0.283312261}, {0.283105493}, {0.2832537},
            {0.283541948}, {0.283633173}, {0.283465385}, {0.283389926},
            {0.283438861}, {0.283398449},
        });

        RealMatrix eulerZ = new Array2DRowRealMatrix(new double [][] {
            {-6.124073982}, {-6.123942852}, {-6.12383604}, {-6.123743057},
            {-6.123628139}, {-6.12349081}, {-6.123329639}, {-6.12319231},
            {-6.123130322}, {-6.123102665}, {-6.123082161}, {-6.123067856},
            {-6.123089314}, {-6.123165131}, {-6.123251438}, {-6.12330246},
            {-6.123318672}, {-6.123353004}, {-6.12339592}, {-6.123435974},
            {-6.123426914}, {-6.123328686}, {-6.123205662}, {-6.123147964},
            {-6.123155594}, {-6.123202801},
        });


        // Create Kalman filter instance
        KalmanFilterExample kalmanFilter = new KalmanFilterExample(deltaPx, deltaPy, deltaPxv, deltaPyv, delta_x, delta_y, delta_xv, delta_yv, delta_t, H, C, accelerations, eulerX, eulerY, eulerZ);


        System.out.println(accelerations.getColumn(0).length + " " + eulerY.getColumn(0).length);

        // Update the Kalman filter with each acceleration
        for (int i = 0; i < accelerations.getColumn(0).length - 2; ++i) {
            System.out.println("Filtered state: [" + kalmanFilter.getState()[0] + ", " + kalmanFilter.getState()[1] + "]");
            kalmanFilter.update();
        }

        kalmanFilter.plotResults();

    }

}
