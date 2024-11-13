package org.example;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.stat.Summarizer;
import org.apache.spark.sql.*;
import org.apache.spark.sql.functions.*;
import org.apache.spark.sql.types.*;
import org.apache.spark.ml.linalg.Matrix;
import org.apache.spark.ml.stat.Correlation;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.VectorAssembler;
import org.knowm.xchart.*;

import org.knowm.xchart.SwingWrapper;

import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("DataPreprocessing")
                .master("local[*]")
                .config("spark.driver.host", "192.168.1.8")
                .config("spark.executor.cores", "4")
                .config("spark.executor.memory", "4g")
                .config("spark.ui.auth.enabled", "true")
                .config("spark.ui.auth.secret", "your_secret_key")
                .config("spark.driver.extraJavaOptions", "-Dsun.reflect.debugModuleAccessChecks=true")
                .config("spark.driver.extraJavaOptions", "--illegal-access=permit")
                .getOrCreate();

       Dataset<Row> df = spark.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("src/main/resources/data_imbalance_loan.csv");

        df = df.drop("ID"); //drop ID

        //Correlation matrix
        String[] numericCols = {"Age", "Experience", "Income", "CCAvg", "Mortgage"};
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(numericCols)
                .setOutputCol("features");

        Dataset<Row> featureDf = assembler.transform(df).select("features");
        Matrix correlationMatrix = Correlation.corr(featureDf, "features").head().getAs(0);
        System.out.println("Correlation Matrix:");

        int columnWidth = 15;
        System.out.printf("%-" + columnWidth + "s", "");
        for (String col : numericCols) {
            System.out.printf("%-" + columnWidth + "s", col);
        }
        System.out.println();
        System.out.println("-".repeat(columnWidth * (numericCols.length + 1)));

        for (int i = 0; i < correlationMatrix.numRows(); i++) {
            System.out.printf("%-" + columnWidth + "s", numericCols[i]);
            for (int j = 0; j < correlationMatrix.numCols(); j++) {
                System.out.printf("%-" + columnWidth + ".2f", correlationMatrix.apply(i, j));
            }
            System.out.println();
        }

        //Check missing values & duplicated values
        System.out.println("\nCheck Missing value");
        for (String col : df.columns()) {
            df.select(functions.sum(functions.when(df.col(col).isNull(), 1).otherwise(0)).alias(col + "_missing_count"))
                    .show();
        }

        System.out.println("\nCheck Duplicated value");
        Column[] columns = new Column[df.columns().length];
        for (int i = 0; i < df.columns().length; i++) {
            columns[i] = functions.col(df.columns()[i]);
        }
        long duplicateCount = df.groupBy(columns)
                .count()
                .filter("count > 1")
                .count();
        System.out.println("Number of duplicates: " + duplicateCount);
        df = df.dropDuplicates(); //Drop duplicated values

        //Print info
        System.out.println("\nSchema Information:");
        df.printSchema();

        //Describe stats
        System.out.println("\nDescribe statistics:");
        Dataset<Row> stats = df.describe();
        stats.show();

        //Boxplot for ZIP Code
        List<Double> zipCodeValues = df.select("ZIP Code")
                .as(Encoders.DOUBLE())
                .collectAsList();

        BoxChart boxChart = new BoxChartBuilder()
                .width(800)
                .height(200)
                .title("Distribution Of ZIP Code")
                .xAxisTitle("ZIP Code")
                .yAxisTitle("Values")
                .build();

        boxChart.addSeries("ZIP Code Distribution", zipCodeValues);

        new SwingWrapper<>(boxChart).displayChart();

        // Drop noise
        df = df.filter("`ZIP Code` >= 20000");
        df = df.withColumn("index", functions.monotonically_increasing_id());

        //In 'Experience', it may include negative values (min = -3) so we use abs function to turn them in to positive numbers.
        df = df.withColumn("Experience", functions.abs(df.col("Experience")));

        //Drop Outlier
        double mean = df.select(functions.avg("Mortgage")).first().getDouble(0);
        double stddev = df.select(functions.stddev("Mortgage")).first().getDouble(0);
        Dataset<Row> zScoreDf = df.withColumn("zscore", (functions.col("Mortgage").minus(mean)).divide(stddev));
        long outlierCount = zScoreDf.filter(functions.abs(zScoreDf.col("zscore")).gt(3)).count();
        System.out.println("Number of outliers (Z-score > 3): " + outlierCount);
        Dataset<Row> cleanedDf = zScoreDf.filter(functions.abs(zScoreDf.col("zscore")).leq(3));
        cleanedDf.show();

        //Feature Transformation
        //In the dataset, CCAVG represents average monthly credit card spending, but Income represents the amount of annual income. To make the units of the features equal, we convert average monthly credit card spending to annual:
        df = df.withColumn("CCAvg", functions.col("CCAvg").multiply(12));

        spark.stop();
    }
}

