package org.example;

import org.jfree.data.xy.XYSeriesCollection;
import weka.core.Instances;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import static org.example.Data.*;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;

public class RandomForrest {
    private Instances df;
    private String targetCol;
    private int classIndex;


    public RandomForrest(Instances df, String targetCol) throws Exception {
        this.df = df;
        this.targetCol = targetCol;
        this.classIndex = getAttributeIndex(targetCol, df);
    }

    public void trainRF() throws Exception {
        // Start time
        long startTime = System.currentTimeMillis();

        // Split the dataset into training and testing sets for 10-fold cross-validation
        Instances[][] splits = splitTrainAndTest(df, 10, 42, classIndex);
        Evaluation[] evals = new Evaluation[10];

        double totalAccuracy = 0;
        double totalMacroPrecision = 0;
        double totalMacroRecall = 0;
        double totalMacroF1 = 0;
        double totalF1Class1 = 0;

        // Loop through each fold for evaluation
        for (int fold = 0; fold < 10; fold++) {
            Instances trainingSet = splits[fold][0];
            Instances testingSet = splits[fold][1];

            // Train RandomForest classifier
            Classifier classifier = new RandomForest();
            classifier.buildClassifier(trainingSet);

            // Evaluate the classifier
            evals[fold] = new Evaluation(trainingSet);
            evals[fold].evaluateModel(classifier, testingSet);

            // Print evaluation results for each class
            System.out.println("Fold | Class | Precision | Recall | F1 Score");
            for (int i = 0; i < trainingSet.numClasses(); i++) {
                System.out.printf("%-5d| %-6d| %-9.5f| %-7.5f| %-9.5f\n",
                        fold + 1, i, evals[fold].precision(i), evals[fold].recall(i), evals[fold].fMeasure(i));
            }

            // Calculate macro metrics for the current fold
            double macroPrecision = 0, macroRecall = 0, macroF1 = 0;
            for (int i = 0; i < trainingSet.numClasses(); i++) {
                macroPrecision += evals[fold].precision(i);
                macroRecall += evals[fold].recall(i);
                macroF1 += evals[fold].fMeasure(i);
            }
            macroPrecision /= trainingSet.numClasses();
            macroRecall /= trainingSet.numClasses();
            macroF1 /= trainingSet.numClasses();

            // Accumulate macro metrics
            totalMacroPrecision += macroPrecision;
            totalMacroRecall += macroRecall;
            totalMacroF1 += macroF1;

            // Accumulate F1-score for class 1
            totalF1Class1 += evals[fold].fMeasure(1);

            // Accumulate accuracy
            totalAccuracy += evals[fold].pctCorrect() / 100.0;

            // Print fold-level macro metrics
            System.out.printf("     | Macro    | %-9.5f| %-7.5f| %-9.5f\n", macroPrecision, macroRecall, macroF1);
            System.out.printf("     | Accuracy | %-9.5f\n", evals[fold].pctCorrect() / 100.0);
        }
        // End time
        long endTime = System.currentTimeMillis();
        long runtime = endTime - startTime;
        System.out.println("\nTotal Runtime: " + runtime + " milliseconds");

        // Calculate overall metrics
        double finalAccuracy = totalAccuracy / 10;
        double finalMacroPrecision = totalMacroPrecision / 10;
        double finalMacroRecall = totalMacroRecall / 10;
        double finalMacroF1 = totalMacroF1 / 10;
        double finalF1Class1 = totalF1Class1 / 10;

        // Print overall metrics
        System.out.printf("\nFinal Metrics over %d folds:\n", 10);
        System.out.printf("Overall Accuracy: %-9.5f\n", finalAccuracy);
        System.out.printf("Overall Macro Precision: %-9.5f\n", finalMacroPrecision);
        System.out.printf("Overall Macro Recall: %-9.5f\n", finalMacroRecall);
        System.out.printf("Overall Macro F1 Score: %-9.5f\n", finalMacroF1);
        System.out.printf("Overall F1 Score for Class 1: %-9.5f\n", finalF1Class1);

    }


}
