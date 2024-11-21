package org.example;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.*;
public class DataProcessor {
    public static Dataset<Row> handleMissingValues(Dataset<Row> df) {
        System.out.println("\nCheck Missing value");
        for (String col : df.columns()) {
            df.select(functions.sum(functions.when(df.col(col).isNull(), 1).otherwise(0)).alias(col + "_missing_count"))
                    .show();
        }
        return df;
    }

    public static Dataset<Row> dropDuplicates(Dataset<Row> df) {
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
        return df.dropDuplicates();
    }

    public static Dataset<Row> filterZipCode(Dataset<Row> df) {
        return df.filter("`ZIP Code` >= 20000");
    }

    public static Dataset<Row> correctNegativeValues(Dataset<Row> df) {
        return df.withColumn("Experience", functions.abs(df.col("Experience")));
    }

    public static Dataset<Row> dropOutliers(Dataset<Row> df, String column) {
        double mean = df.select(functions.avg(column)).first().getDouble(0);
        double stddev = df.select(functions.stddev(column)).first().getDouble(0);
        Dataset<Row> zScoreDf = df.withColumn("zscore", (functions.col(column).minus(mean)).divide(stddev));
        return zScoreDf.filter(functions.abs(zScoreDf.col("zscore")).leq(3));
    }
}

