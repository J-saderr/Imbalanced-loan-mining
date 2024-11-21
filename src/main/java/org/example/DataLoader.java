package org.example;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
public class DataLoader {
    public static Dataset<Row> loadData(SparkSession spark, String filePath) {
        return spark.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load(filePath)
                .drop("ID");
    }
}

