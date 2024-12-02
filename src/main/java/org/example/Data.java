package org.example;

import org.apache.spark.sql.*;
import org.apache.spark.sql.functions;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Standardize;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.abs;

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

    public static Instances numToNor(Instances df, String cols) throws Exception {
        // Split the comma-separated column names into a list
        List<String> colNames = Arrays.asList(cols.split(",\\s*"));

        // Find the indices of the specified columns
        List<Integer> indices = new ArrayList<>();
        for (String col : colNames) {
            Attribute attr = df.attribute(col);
            if (attr == null) {
                throw new IllegalArgumentException("Column name not found: " + col);
            }
            if (!attr.isNumeric()) {
                throw new IllegalArgumentException("Column is not numeric: " + col);
            }
            indices.add(attr.index() + 1); // Convert 0-based index to 1-based index
        }

        // Convert the indices list to a comma-separated string
        String indicesStr = indices.toString().replaceAll("[\\[\\] ]", "");

        // Initialize and configure the NumericToNominal filter
        NumericToNominal numToNo = new NumericToNominal();
        numToNo.setAttributeIndices(indicesStr); // Set the indices of attributes to convert
        numToNo.setInputFormat(df); // Configure the filter with the input dataset

        // Apply the filter and return the modified dataset
        return Filter.useFilter(df, numToNo);
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

    //Method standardized data
    public static Instances scaleAttributes(Instances data) throws Exception {
        Standardize standardizeFilter = new Standardize();
        standardizeFilter.setInputFormat(data);
        Instances scaledData = Filter.useFilter(data, standardizeFilter);

        return scaledData;
    }

    public static int countClassInstances(Instances df, int classValue) {
        int count = 0;
        for (int i = 0; i < df.numInstances(); i++) {
            if (df.instance(i).classValue() == classValue) {
                count++;
            }
        }
        return count;
    }

    public static Instances applySMOTE(Instances df, int classIndex) throws Exception {
        df.setClassIndex(classIndex);

        int class0Count = countClassInstances(df, 0);
        int class1Count = countClassInstances(df, 1);
        int difference = abs(class1Count - class0Count);

        // Apply SMOTE iteratively to balance the classes
        while (difference > 10) {
            SMOTE smote = new SMOTE();

            // Target the class with fewer instances to match the larger class
            if (class0Count < class1Count) {
                smote.setPercentage((class1Count - class0Count) * 100 / class0Count);
            } else {
                smote.setPercentage((class0Count - class1Count) * 100 / class1Count);
            }
            smote.setInputFormat(df);  // Define the input format for SMOTE
            df = Filter.useFilter(df, smote);  // Apply SMOTE

            class0Count = countClassInstances(df, 0);
            class1Count = countClassInstances(df, 1);
            difference = abs(class1Count - class0Count);
        }
        return df;
    }


    public static Instances[][] splitTrainAndTest(Instances df, int folds, int randomSeed, int classIndex) throws Exception {
        df.randomize(new Random(randomSeed));
        df.setClassIndex(classIndex);

        // Initialize splits array: [fold][0 for train, 1 for test]
        Instances[][] splits = new Instances[folds][2];

        for (int fold = 0; fold < folds; fold++) {
            // Split the nominal dataset into training and testing sets for the current fold
            Instances trainingSet = df.trainCV(folds, fold);
            Instances testingSet = df.testCV(folds, fold);

            // Ensure the class attribute is set for the training and testing sets
            trainingSet.setClassIndex(df.classIndex());
            testingSet.setClassIndex(df.classIndex());

            // Save the current fold's training and testing datasets
            splits[fold][0] = trainingSet;
            splits[fold][1] = testingSet;

        }
        return splits;
    }
}
