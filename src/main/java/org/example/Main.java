package org.example;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.*;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instances;

import static org.example.Data.*;

public class Main {

    public static void main(String[] args) throws Exception {

        Data df = new Data("src/main/resources/data_imbalance_loan.csv");
        Dataset<Row> processedDf = df.dropCols("ID", "Experience");

        // Convert Spark Dataset<Row> to Weka Instances
        Instances wekaDf = WEKAdata(processedDf);

        //Set class index
        int classIndex = getAttributeIndex("Personal Loan", wekaDf);


        //Split train and test
        Instances[][] splits = splitTrainAndTest(wekaDf, 10, 42, classIndex);

        for (int fold = 0; fold < 10; fold++) {
            Instances trainingSet = splits[fold][0];
            Instances testingSet = splits[fold][1];

            // Now you can train a classifier on trainingSet and evaluate on testingSet
            Classifier classifier = new RandomForest(); // Or any other classifier
            classifier.buildClassifier(trainingSet);

            // Evaluate model on the testing set
            Evaluation eval = new Evaluation(trainingSet);
            eval.evaluateModel(classifier, testingSet);

            // Print evaluation results for each fold
            System.out.println("Fold | Class | Precision | Recall | F1 Score");

                // Iterate over each class
            for (int i = 0; i < trainingSet.numClasses(); i++) {
            System.out.printf("%-5d| %-6d| %-9.2f| %-7.2f| %-9.2f\n", fold + 1, i, eval.precision(i), eval.recall(i), eval.fMeasure(i));
            }
        }



//        SparkSession spark = SparkSession.builder()
//                .appName("DataPreprocessing")
//                .master("local[*]")
//                .config("spark.driver.host", "192.168.1.8")
//                .config("spark.driver.bindAddress", "127.0.0.1")
//                .config("spark.driver.port", "4040")
//                .config("spark.executor.cores", "4")
//                .config("spark.executor.memory", "4g")
//                .config("spark.ui.auth.enabled", "true")
//                .config("spark.ui.auth.secret", "secret_key")
//                .config("spark.driver.extraJavaOptions", "-Dsun.reflect.debugModuleAccessChecks=true --illegal-access=permit")
//                .getOrCreate();
//
//       Dataset<Row> df = spark.read().format("csv")
//                .option("header", "true")
//                .option("inferSchema", "true")
//                .load("src/main/resources/data_imbalance_loan.csv");
//
//        df = df.drop("ID"); //drop ID
//
//        //Correlation matrix
//        String[] numericCols = {"Age", "Experience", "Income", "CCAvg", "Mortgage"};
//        VectorAssembler assembler = new VectorAssembler()
//                .setInputCols(numericCols)
//                .setOutputCol("features");
//
//        Dataset<Row> featureDf = assembler.transform(df).select("features");
//        Matrix correlationMatrix = Correlation.corr(featureDf, "features").head().getAs(0);
//        System.out.println("Correlation Matrix:");
//        int columnWidth = 15;
//        System.out.printf("%-" + columnWidth + "s", "");
//        for (String col : numericCols) {
//            System.out.printf("%-" + columnWidth + "s", col);
//        }
//        System.out.println();
//        System.out.println("-".repeat(columnWidth * (numericCols.length + 1)));
//
//        for (int i = 0; i < correlationMatrix.numRows(); i++) {
//            System.out.printf("%-" + columnWidth + "s", numericCols[i]);
//            for (int j = 0; j < correlationMatrix.numCols(); j++) {
//                System.out.printf("%-" + columnWidth + ".2f", correlationMatrix.apply(i, j));
//            }
//            System.out.println();
//        }
//
//        //Check missing values & duplicated values
//        System.out.println("\nCheck Missing value");
//        for (String col : df.columns()) {
//            df.select(functions.sum(functions.when(df.col(col).isNull(), 1).otherwise(0)).alias(col + "_missing_count"))
//                    .show();
//        }
//
//        System.out.println("\nCheck Duplicated value");
//        Column[] columns = new Column[df.columns().length];
//        for (int i = 0; i < df.columns().length; i++) {
//            columns[i] = functions.col(df.columns()[i]);
//        }
//        long duplicateCount = df.groupBy(columns)
//                .count()
//                .filter("count > 1")
//                .count();
//        System.out.println("Number of duplicates: " + duplicateCount);
//        df = df.dropDuplicates(); //Drop duplicated values
//
//        //Print info
//        System.out.println("\nSchema Information:");
//        df.printSchema();
//
//        //Describe stats
//        System.out.println("\nDescribe statistics:");
//        Dataset<Row> stats = df.describe();
//        stats.show();
//
//        List<Double> zipCodeValues = df.select("ZIP Code")
//                .as(Encoders.DOUBLE())
//                .collectAsList();
//
//        // Create a BoxChart using XChart
//        BoxChart boxChart = new BoxChartBuilder()
//                .width(800)
//                .height(400) // Adjust height as needed
//                .title("Distribution Of ZIP Code")
//                .xAxisTitle("ZIP Code")
//                .yAxisTitle("Values")
//                .build();
//
//        // Add the ZIP Code data as a series
//        boxChart.addSeries("ZIP Code Distribution", zipCodeValues);
//
//        // Create a ChartPanel for displaying the BoxChart
//        XChartPanel<BoxChart> chartPanel1 = new XChartPanel<>(boxChart);
//
//        // Create a JFrame to hold the chart
//        JFrame frame = new JFrame("BoxPlot for ZIP Code");
//        frame.setLayout(new java.awt.BorderLayout());
//        frame.setSize(800, 600); // Set the size of the frame
//        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//
//        // Add the chart panel to the JFrame
//        frame.add(chartPanel1, java.awt.BorderLayout.CENTER);
//
//        // Make the JFrame visible
//        frame.setVisible(true);

//        // Drop noise
//        df = df.filter("`ZIP Code` >= 20000");
//        df = df.withColumn("index", functions.monotonically_increasing_id());
//
//        //In 'Experience', it may include negative values (min = -3) so we use abs function to turn them in to positive numbers.
//        df = df.withColumn("Experience", functions.abs(df.col("Experience")));
//
//        //Drop Outlier
//        double mean = df.select(functions.avg("Mortgage")).first().getDouble(0);
//        double stddev = df.select(functions.stddev("Mortgage")).first().getDouble(0);
//        Dataset<Row> zScoreDf = df.withColumn("zscore", (functions.col("Mortgage").minus(mean)).divide(stddev));
//        long outlierCount = zScoreDf.filter(functions.abs(zScoreDf.col("zscore")).gt(3)).count();
//        System.out.println("Number of outliers (Z-score > 3): " + outlierCount);
//        Dataset<Row> cleanedDf = zScoreDf.filter(functions.abs(zScoreDf.col("zscore")).leq(3));
//        cleanedDf.show();
//
//        //Feature Transformation
//        //In the dataset, CCAVG represents average monthly credit card spending, but Income represents the amount of annual income. To make the units of the features equal, we convert average monthly credit card spending to annual:
//        df = df.withColumn("CCAvg", functions.col("CCAvg").multiply(12));
//
//        //Bivariate Analysis
//        List<String> numericalFeatures = List.of("CCAvg", "Income", "Mortgage", "Age", "Experience");
//        List<String> categoricalFeatures = List.of("Family", "Education", "Securities Account", "CD Account", "Online", "CreditCard");
//
//        JFrame frameNumerical = new JFrame("Numerical Features vs Target Distribution");
//        frameNumerical.setLayout(new GridLayout(5, 2));
//        frameNumerical.setSize(1600, 1000);
//        frameNumerical.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//
//        for (String feature : numericalFeatures) {
//            DefaultCategoryDataset barDataset = createBarChartDataset(df, feature);
//            JFreeChart barChart = createBarChart(feature, barDataset);
//            ChartPanel barPanel = new ChartPanel(barChart);
//            frameNumerical.add(barPanel);
//
//            XYSeriesCollection kdeDataset = createKdeDataset(df, feature);
//            JFreeChart kdeChart = createKdeChart(feature, kdeDataset);
//            ChartPanel kdePanel = new ChartPanel(kdeChart);
//            frameNumerical.add(kdePanel);
//        }
//
//        frameNumerical.setVisible(true);
//
//        JFrame frameCategorical = new JFrame("Categorical Features vs Target Distribution");
//        frameCategorical.setLayout(new GridLayout(2, 3));
//        frameCategorical.setSize(1200, 800);
//        frameCategorical.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//
//        for (String feature : categoricalFeatures) {
//            Dataset<Row> featureData = df.groupBy(feature, "Personal Loan")
//                    .count()
//                    .groupBy(feature)
//                    .pivot("Personal Loan")
//                    .agg(functions.sum("count"))
//                    .na()
//                    .fill(0);
//
//            DefaultCategoryDataset dataset = createDatasetFromSpark(featureData, feature);
//            JFreeChart chart = createStackedBarChart(feature, dataset);
//            ChartPanel chartPanel = new ChartPanel(chart);
//            frameCategorical.add(chartPanel);
//        }
//
//        frameCategorical.setVisible(true);
//
//        df = df.drop("Experience");
//
//        //Train & Test
//        Dataset<Row> X = df.drop("Personal Loan");
//        Dataset<Row> y = df.select("Personal Loan");
//
//        Dataset<Row>[] splits = df.randomSplit(new double[]{0.8, 0.2}, 1234L);
//        Dataset<Row> trainData = splits[0];
//        Dataset<Row> testData = splits[1];
//        spark.stop();
//    }

    //bar chart categorical
//    private static DefaultCategoryDataset createDatasetFromSpark(Dataset<Row> sparkDataset, String feature) {
//        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
//
//        for (Row row : sparkDataset.collectAsList()) {
//            String category = String.valueOf(row.get(0));
//            double loan0Count = row.getAs(1) != null ? ((Number) row.getAs(1)).doubleValue() : 0.0;
//            double loan1Count = row.getAs(2) != null ? ((Number) row.getAs(2)).doubleValue() : 0.0;
//
//            double total = loan0Count + loan1Count;
//            double loan0Proportion = total > 0 ? loan0Count / total : 0.0;
//            double loan1Proportion = total > 0 ? loan1Count / total : 0.0;
//
//            dataset.addValue(loan0Proportion, "Loan 0", category);
//            dataset.addValue(loan1Proportion, "Loan 1", category);
//        }
//
//        return dataset;
//    }
//    private static JFreeChart createStackedBarChart(String feature, DefaultCategoryDataset dataset) {
//        JFreeChart chart = ChartFactory.createStackedBarChart(
//                "Feature: " + feature + " vs Personal Loan",
//                feature,
//                "Proportion",
//                dataset
//        );
//
//        CategoryPlot plot = (CategoryPlot) chart.getPlot();
//        StackedBarRenderer renderer = new StackedBarRenderer();
//        plot.setRenderer(renderer);
//
//        return chart;
//    }

    //Bar Chart numerical
//    private static DefaultCategoryDataset createBarChartDataset(Dataset<Row> df, String feature) {
//        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
//
//        Dataset<Row> avgData = df.groupBy("Personal Loan")
//                .agg(functions.avg(feature).alias("average"));
//
//        avgData.collectAsList().forEach(row -> {
//            int loanCategory = row.getInt(0);
//            double averageValue = row.getDouble(1);
//            dataset.addValue(averageValue, "Personal Loan " + loanCategory, feature);
//        });
//
//        return dataset;
//    }

//    private static JFreeChart createBarChart(String feature, DefaultCategoryDataset dataset) {
//        return ChartFactory.createBarChart(
//                "Bar Chart: " + feature + " vs Personal Loan",
//                "Target",
//                "Average Value",
//                dataset,
//                PlotOrientation.VERTICAL,
//                true,
//                true,
//                false
//        );
//    }

    //KDE numerical
//    private static XYSeriesCollection createKdeDataset(Dataset<Row> df, String feature) {
//        XYSeries series0 = new XYSeries("Loan 0");
//        XYSeries series1 = new XYSeries("Loan 1");
//
//        List<Row> loan0Data = df.filter("`Personal Loan` = 0").select(feature).collectAsList();
//        List<Row> loan1Data = df.filter("`Personal Loan` = 1").select(feature).collectAsList();
//
//        Map<Double, Double> kdeLoan0 = computeKDE(loan0Data);
//        Map<Double, Double> kdeLoan1 = computeKDE(loan1Data);
//
//        kdeLoan0.forEach((x, y) -> series0.add(x, y));
//        kdeLoan1.forEach((x, y) -> series1.add(x, y));
//
//        XYSeriesCollection dataset = new XYSeriesCollection();
//        dataset.addSeries(series0);
//        dataset.addSeries(series1);
//
//        return dataset;
//    }

//    private static Map<Double, Double> computeKDE(List<Row> data) {
//        Map<Double, Double> kde = new TreeMap<>();
//        double bandwidth = 1.0;
//        double kernelScale = 1.0 / (Math.sqrt(2 * Math.PI) * bandwidth);
//
//        for (Row row : data) {
//            double value;
//            if (row.get(0) instanceof Integer) {
//                value = row.getInt(0);
//            } else if (row.get(0) instanceof Double) {
//                value = row.getDouble(0);
//            } else {
//                continue;
//            }
//
//            for (double x = value - 5; x <= value + 5; x += 0.1) {
//                double kernelValue = kernelScale * Math.exp(-0.5 * Math.pow((x - value) / bandwidth, 2));
//                kde.put(x, kde.getOrDefault(x, 0.0) + kernelValue);
//            }
//        }
//
//        return kde;
//    }
//    private static JFreeChart createKdeChart(String feature, XYSeriesCollection dataset) {
//        JFreeChart chart = ChartFactory.createXYLineChart(
//                "KDE Plot: " + feature + " vs Personal Loan",
//                feature,
//                "Density",
//                dataset,
//                PlotOrientation.VERTICAL,
//                true,
//                true,
//                false
//        );
//
//        XYPlot plot = (XYPlot) chart.getPlot();
//        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
//        renderer.setSeriesPaint(0, Color.BLUE); // Loan 0 (Personal Loan = 0)
//        renderer.setSeriesPaint(1, Color.RED); // Loan 1 (Personal Loan = 1)
//        plot.setRenderer(renderer);
//
//        NumberAxis yAxis = (NumberAxis) plot.getRangeAxis();
//        yAxis.setAutoRangeIncludesZero(false);
//
//        return chart;
    }
}

