package org.example;

import weka.core.Instances;

import static org.example.Data.getAttributeIndex;
import static org.example.Data.splitTrainAndTest;
import static org.example.Data.scaleAttributes;
import weka.classifiers.lazy.IBk;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.EuclideanDistance;
import weka.core.MinkowskiDistance;
import weka.classifiers.Evaluation;

public class kNN {
    private Instances df;
    private int classIndex;
    private String targetCol;

    public kNN(Instances df, String targetCol) throws Exception {
        this.df = df;
        this.targetCol = targetCol;
        this.classIndex = getAttributeIndex(targetCol, df);
    }

    public void trainKNN() throws Exception {
        // Perform feature selection if needed
        df = scaleAttributes(df);

        // Split the dataset into training and testing sets for 10-fold cross-validation
        Instances[][] splits = splitTrainAndTest(df, 10, 42, classIndex);
        Evaluation[] evals = new Evaluation[10];

        double totalAccuracy = 0;
        double totalF1score = 0;

        // Loop through each fold for evaluation
        for (int fold = 0; fold < 10; fold++) {
            Instances trainingSet = splits[fold][0];
            Instances testingSet = splits[fold][1];

            // Train kNN classifier (using IBk)
            IBk knn = new IBk();
            knn.setKNN(3);

            // Set Minkowski distance as the distance metric
            LinearNNSearch search = new LinearNNSearch();
            MinkowskiDistance minkowskiDistance = new MinkowskiDistance();
            minkowskiDistance.setOrder(3);
            search.setDistanceFunction(minkowskiDistance);
            knn.setNearestNeighbourSearchAlgorithm(search);

            // Train the kNN classifier
            knn.buildClassifier(trainingSet);

            // Evaluate the classifier
            evals[fold] = new Evaluation(trainingSet);
            evals[fold].evaluateModel(knn, testingSet);

            // Print evaluation metrics
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

            // Calculate macro and weighted averages
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

            // Print average metrics
            System.out.printf("     | Macro    | %-9.5f| %-7.5f| %-9.5f\n", macroPrecision, macroRecall, macroF1);
            System.out.printf("     | Weighted | %-9.5f| %-7.5f| %-9.5f\n", weightedPrecision, weightedRecall, weightedF1);
            System.out.printf("     | Accuracy | %-9.5f\n", evals[fold].pctCorrect() / 100.0);
        }
        double finalAccuracy = totalAccuracy / 10;
        System.out.printf("\nFinal Accuracy over %d folds: %-9.5f\n", 10, finalAccuracy);

        double finalF1score = totalF1score/10;
        System.out.printf("\nFinal F1 Score over %d folds: %-9.5f\n", 10, finalF1score);
    }
}
