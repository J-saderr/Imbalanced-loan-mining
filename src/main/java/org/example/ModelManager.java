package org.example;

import org.apache.spark.sql.*;
public class ModelManager {
    public static Dataset<Row>[] splitData(Dataset<Row> df, double trainRatio) {
        return df.randomSplit(new double[]{trainRatio, 1 - trainRatio}, 1234L);
    }
}

