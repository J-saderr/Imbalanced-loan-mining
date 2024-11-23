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
        // Split train and test
        Instances[][] splits = splitTrainAndTest(df, 10, 42, classIndex);
        Evaluation[] evals = new Evaluation[10];
        //Final overall scores
        double totalAccuracy = 0;
        double totalF1score = 0;

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



            // Print evaluation results
            System.out.println("Fold | Class | Precision | Recall | F1 Score");
            for (int i = 0; i < trainingSet.numClasses(); i++) {
                System.out.printf("%-5d| %-6d| %-9.5f| %-7.5f| %-9.5f\n",
                        fold + 1, i, evals[fold].precision(i), evals[fold].recall(i), evals[fold].fMeasure(i));
            }
            double foldF1 = 0;
            for (int i = 0; i < trainingSet.numClasses(); i++) {
                foldF1 += evals[fold].fMeasure(i); // Sum the F1 scores for each class
            }

            totalF1score += foldF1 / trainingSet.numClasses();

        // Calculate macro and weighted averages for precision, recall, and F1 score
            double macroPrecision = 0, macroRecall = 0, macroF1 = 0;
            double weightedPrecision = 0, weightedRecall = 0, weightedF1 = 0;
            int[] classCounts = trainingSet.attributeStats(classIndex).nominalCounts;
            double totalInstances = trainingSet.numInstances();

            for (int i = 0; i < trainingSet.numClasses(); i++) {
                double classPrecision = evals[fold].precision(i);
                double classRecall = evals[fold].recall(i);
                double classF1 = evals[fold].fMeasure(i);

                macroPrecision += classPrecision;
                macroRecall += classRecall;
                macroF1 += classF1;

                double weight = classCounts[i] / totalInstances;
                weightedPrecision += classPrecision * weight;
                weightedRecall += classRecall * weight;
                weightedF1 += classF1 * weight;
            }

            macroPrecision /= trainingSet.numClasses();
            macroRecall /= trainingSet.numClasses();
            macroF1 /= trainingSet.numClasses();

            // Accuracy for the current fold
            double foldAccuracy = evals[fold].pctCorrect() / 100.0;
            totalAccuracy += foldAccuracy;

            System.out.printf("     | Macro  | %-9.5f| %-7.5f| %-9.5f\n", macroPrecision, macroRecall, macroF1);
            System.out.printf("     | Weighted | %-9.5f| %-7.5f| %-9.5f\n", weightedPrecision, weightedRecall, weightedF1);
            System.out.printf("     | Accuracy | %-9.5f\n", evals[fold].pctCorrect() / 100.0);
        }
        double finalAccuracy = totalAccuracy /10;
        System.out.printf("\nFinal Accuracy over %d folds: %-9.5f\n", 10, finalAccuracy);

        double finalF1score = totalF1score/10;
        System.out.printf("\nFinal F1 Score over %d folds: %-9.5f\n", 10, finalF1score);
    }
}
