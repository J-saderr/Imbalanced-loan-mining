package org.example;
import org.apache.spark.sql.SparkSession;
public class SparkSessionManager {
    public static SparkSession createSession() {
        return SparkSession.builder()
                .appName("DataPreprocessing")
                .master("local[*]")
                .config("spark.driver.host", "192.168.1.8")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .config("spark.driver.port", "4040")
                .config("spark.executor.cores", "4")
                .config("spark.executor.memory", "4g")
                .config("spark.ui.auth.enabled", "true")
                .config("spark.ui.auth.secret", "secret_key")
                .config("spark.driver.extraJavaOptions", "-Dsun.reflect.debugModuleAccessChecks=true --illegal-access=permit")
                .getOrCreate();
    }
}

