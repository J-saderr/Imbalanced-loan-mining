package org.example;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

import java.util.*;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import static org.example.Data.*;
import static org.example.Visualization.plotHistogram;

public class Main {

    public static void main(String[] args) throws Exception {
        SparkSession spark = SparkSessionManager.createSession();

        Dataset<Row> df = DataLoader.loadData(spark, "src/main/resources/data_imbalance_loan.csv");

        //drop ID
        df = df.drop("ID");

        //Correlation matrix
        CorrelationMatrix.CorrelationMatrixs(df);

        //Print info
        System.out.println("\nSchema Information:");
        df.printSchema();

        //Describe stats
        System.out.println("\nDescribe statistics:");
        Dataset<Row> stats = df.describe();
        stats.show();

        //Histogram
        Visualization.plotHistogram(df, "Age", "Experience", "Income", "CCAvg", "Mortgage");

        //Check missing values & duplicated values
        df = DataProcessor.handleMissingValues(df);
        df = DataProcessor.dropDuplicates(df);

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


        //Data for Decision Tree
        Dataset<Row> DTdata = df.drop("Online");
        Instances DTdf = WEKAdata(DTdata);
        DTdf = numToNor(DTdf,"Securities Account, CD Account, CreditCard, Personal Loan");

        //Get class index
        int classIndex1 = getAttributeIndex("Personal Loan",DTdf);

        Instances smotedDTdf = applySMOTE(DTdf, classIndex1);

        //Decision Tree model
        DecisionTree classifier2 = new DecisionTree(smotedDTdf,"Personal Loan");
        classifier2.trainDT();

        //Data for Random Forrest
        Dataset<Row> RFdata = df.drop("ID");
        Instances RFdf = WEKAdata(RFdata);
        RFdf = numToNor(RFdf,"Securities Account, CD Account, Online, CreditCard, Personal Loan");
        int classIndex2 = getAttributeIndex("Personal Loan",RFdf);

        Instances smotedRFdf = applySMOTE(RFdf, classIndex2);

        //Random Forrest model
        RandomForrest rf = new RandomForrest(smotedRFdf, "Personal Loan");
        rf.trainRF();

        spark.stop();
    }
}

