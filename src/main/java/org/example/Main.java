package org.example;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        SparkSession spark = SparkSessionManager.createSession();

        Dataset<Row> df = DataLoader.loadData(spark, "src/main/resources/data_imbalance_loan.csv");

        //drop ID
        df = df.drop("ID");

        //Correlation matrix
        CorrelationMatrix.CorrelationMatrixs(df);

        //Check missing values & duplicated values
        df = DataProcessor.handleMissingValues(df);
        df = DataProcessor.dropDuplicates(df);

        //Print info
        System.out.println("\nSchema Information:");
        df.printSchema();

        //Describe stats
        System.out.println("\nDescribe statistics:");
        Dataset<Row> stats = df.describe();
        stats.show();

        //In 'Experience', it may include negative values (min = -3) so we use abs function to turn them in to positive numbers.
        df = DataProcessor.correctNegativeValues(df);

        //Boxplot
        Visualization.showBoxPlot(df, "ZIP Code");

        // Drop noise
        df = DataProcessor.filterZipCode(df);

        //Drop Outliers
        df = DataProcessor.dropOutliers(df, "Mortgage");

        //Feature Transformation
        //In the dataset, CCAVG represents average monthly credit card spending, but Income represents the amount of annual income. To make the units of the features equal, we convert average monthly credit card spending to annual:
        df = df.withColumn("CCAvg", functions.col("CCAvg").multiply(12));

        //Bivariate Analysis
        //bar chart & KDE chart for numerical
        Visualization.showNumericalFeatureCharts(df, List.of("CCAvg", "Income", "Mortgage", "Age", "Experience"));

        //The distribution of the Experience is very similar to the distribution of Age, as Experience is strongly correlated with Age.
        //Drop Experience
        df = df.drop("Experience");

        //barchart for categorical
        Visualization.showCategoricalFeatureCharts(df, List.of("Family", "Education", "Securities Account", "CD Account", "Online", "CreditCard"));

        //Train and test
        Dataset<Row>[] splits = ModelManager.splitData(df, 0.8);
        Dataset<Row> trainData = splits[0];
        Dataset<Row> testData = splits[1];

        spark.stop();
    }
}