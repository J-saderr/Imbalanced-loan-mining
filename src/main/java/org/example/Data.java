package org.example;

import org.apache.spark.sql.*;
import org.apache.spark.sql.functions;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.NumericToNominal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Data {
    //Convert attribute from Spark dataset to WEKA dataset
    public static ArrayList<Attribute> getAttributes(Dataset<Row> dataset) {
        ArrayList<Attribute> attributes = new ArrayList<>();

        for (String colName : dataset.columns()) {
            if (dataset.schema().apply(colName).dataType().typeName().equals("double") ||
                    dataset.schema().apply(colName).dataType().typeName().equals("integer")) {
                attributes.add(new Attribute(colName)); // Numeric attributes
            } else {
                // For nominal attributes, get the distinct values
                List<String> nominalValues = new ArrayList<>();
                dataset.select(colName).distinct().collectAsList()
                        .forEach(row -> nominalValues.add(row.getString(0)));
                attributes.add(new Attribute(colName, nominalValues)); // Nominal attributes
            }
        }

        return attributes;
    }
    //Get attribute index
    public static int getAttributeIndex(String columnName, Instances wekaInstances) {
        for (int i = 0; i < wekaInstances.numAttributes(); i++) {
            if (wekaInstances.attribute(i).name().equalsIgnoreCase(columnName)) {
                return i;
            }
        }
        return -1; // Not found
    }

    //Convert to WEKA Instances
    public static Instances WEKAdata(Dataset<Row> dataset) {
        // Get attributes from the dataset using the helper method
        ArrayList<Attribute> attributes = getAttributes(dataset);

        // Create Weka Instances object
        Instances wekaInstances = new Instances("Df", attributes, 0);

        // Add rows from Spark Dataset<Row> to Weka Instances
        List<Row> rows = dataset.collectAsList();
        for (Row row : rows) {
            double[] values = new double[row.size()];
            for (int i = 0; i < row.size(); i++) {
                Object value = row.get(i);
                if (value instanceof Number) {
                    values[i] = ((Number) value).doubleValue(); // For numeric values
                } else {
                    values[i] = wekaInstances.attribute(i).indexOfValue(value.toString()); // For nominal values
                }
            }
            wekaInstances.add(new DenseInstance(1.0, values));
        }

        return wekaInstances;
    }



    public static Instances[][] splitTrainAndTest(Instances data, int folds, int randomSeed, int classIndex) throws Exception {

        //Change Target value to nominal data type because SMOTE requires that.
        NumericToNominal numToNo = new NumericToNominal();
        numToNo.setAttributeIndices(String.valueOf(classIndex + 1));
        numToNo.setInputFormat(data);
        Instances df = Filter.useFilter(data,numToNo);
        df.setClassIndex(classIndex);

        //Instances df = convertToNominal(data, classIndex);
        df.randomize(new Random(randomSeed));

        // Initialize splits array: [fold][0 for train, 1 for test]
        Instances[][] splits = new Instances[folds][2];

        for (int fold = 0; fold < folds; fold++) {
            // Split the nominal dataset into training and testing sets for the current fold
            Instances trainingSet = df.trainCV(folds, fold);
            Instances testingSet = df.testCV(folds, fold);

            // Ensure the class attribute is set for the training and testing sets
            trainingSet.setClassIndex(df.classIndex());
            testingSet.setClassIndex(df.classIndex());

            // Apply SMOTE to the training set
            SMOTE smote = new SMOTE();
            smote.setInputFormat(trainingSet);  // Define the input format for SMOTE
            Instances trainingSetSmote = Filter.useFilter(trainingSet, smote);  // Apply SMOTE to the training set

            // Save the current fold's training and testing datasets
            splits[fold][0] = trainingSetSmote;
            splits[fold][1] = testingSet;
        }
        return splits;
    }
}
