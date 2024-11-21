package org.example;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.stat.Correlation;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.linalg.Matrix;
public class CorrelationMatrix {
    public static void CorrelationMatrixs (Dataset<Row> df){
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
    }
}
