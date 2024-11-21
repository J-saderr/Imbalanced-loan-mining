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
    private Dataset<Row> df;
    private Instances wekaDf;
    private String targetCol;
    private int classIndex;


    public RandomForrest(Dataset<Row> df, String targetCol) throws Exception {
        this.df = df;
        this.targetCol = targetCol;

        // Convert Spark DataFrame to Weka Instances
        this.wekaDf = WEKAdata(df);
        this.classIndex = getAttributeIndex(targetCol, wekaDf);
    }

    public void trainRF() throws Exception {
        // Split train and test
        Instances[][] splits = splitTrainAndTest(wekaDf, 10, 42, classIndex);
        Evaluation[] evals = new Evaluation[10];
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
                System.out.printf("%-5d| %-6d| %-9.2f| %-7.2f| %-9.2f\n",
                        fold + 1, i, evals[fold].precision(i), evals[fold].recall(i), evals[fold].fMeasure(i));
            }

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

            System.out.printf("     | Macro  | %-9.2f| %-7.2f| %-9.2f\n", macroPrecision, macroRecall, macroF1);
            System.out.printf("     | Weighted | %-9.2f| %-7.2f| %-9.2f\n", weightedPrecision, weightedRecall, weightedF1);
            System.out.printf("     | Accuracy | %-9.2f\n", evals[fold].pctCorrect() / 100.0);

        }
    }
}
