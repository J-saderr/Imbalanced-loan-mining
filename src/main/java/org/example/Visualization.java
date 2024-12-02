package org.example;
import org.apache.spark.sql.*;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.CombinedDomainXYPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.category.StackedBarRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.knowm.xchart.BoxChart;
import org.knowm.xchart.BoxChartBuilder;
import org.knowm.xchart.XChartPanel;

import javax.swing.*;
import java.awt.*;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class Visualization {
    public static void plotHistogram(Dataset<Row> df, String... columns) {
        JFrame frame = new JFrame("Histograms");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(1200, 800);
        int rows = (int) Math.ceil(columns.length / 2.0);
        JPanel panel = new JPanel(new GridLayout(rows, 2));
        for (String column : columns) {
            List<Double> columnData = df.select(column).as(Encoders.DOUBLE()).collectAsList();
            double[] values = columnData.stream().mapToDouble(Double::doubleValue).toArray();
            HistogramDataset dataset = new HistogramDataset();
            dataset.addSeries(column, values, 50);
            JFreeChart histogram = ChartFactory.createHistogram(
                    "Distribution of " + column,
                    column,
                    "Frequency",
                    dataset
            );
            panel.add(new ChartPanel(histogram));
        }
        frame.add(panel);
        frame.setVisible(true);
    }

    public static void showBoxPlot(Dataset<Row> df, String columnName) {
        List<Double> zipCodeValues = df.select("ZIP Code")
                .as(Encoders.DOUBLE())
                .collectAsList();
        BoxChart boxChart = new BoxChartBuilder()
                .width(800)
                .height(400)
                .title("Distribution Of ZIP Code")
                .xAxisTitle("ZIP Code")
                .yAxisTitle("Values")
                .build();

        boxChart.addSeries("ZIP Code Distribution", zipCodeValues);

        XChartPanel<BoxChart> chartPanel1 = new XChartPanel<>(boxChart);
        JFrame frame = new JFrame("BoxPlot for " + columnName);
        frame.setLayout(new BorderLayout());
        frame.setSize(800, 600);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(chartPanel1, BorderLayout.CENTER);

        frame.setVisible(true);
    }

    public static void showNumericalFeatureCharts(Dataset<Row> df, List<String> numericalFeatures) {
        JFrame frameNumerical = new JFrame("Numerical Features vs Target Distribution");
        frameNumerical.setLayout(new GridLayout(numericalFeatures.size(), 2));
        frameNumerical.setSize(1600, 1000);
        frameNumerical.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        for (String feature : numericalFeatures) {
            DefaultCategoryDataset barDataset = createBarChartDataset(df, feature);
            JFreeChart barChart = createBarChart(feature, barDataset);
            ChartPanel barPanel = new ChartPanel(barChart);
            frameNumerical.add(barPanel);

            XYSeriesCollection loan0Dataset = new XYSeriesCollection(createKdeSeries(df, feature, 0));
            JFreeChart loan0KdeChart = createKdeChart(feature, loan0Dataset, "Loan 0");
            ChartPanel loan0Panel = new ChartPanel(loan0KdeChart);
            frameNumerical.add(loan0Panel);

            // KDE Plot for Loan 1
            XYSeriesCollection loan1Dataset = new XYSeriesCollection(createKdeSeries(df, feature, 1));
            JFreeChart loan1KdeChart = createKdeChart(feature, loan1Dataset, "Loan 1");
            ChartPanel loan1Panel = new ChartPanel(loan1KdeChart);
            frameNumerical.add(loan1Panel);
        }

        frameNumerical.setVisible(true);
    }

    public static void showCategoricalFeatureCharts(Dataset<Row> df, List<String> categoricalFeatures) {
        JFrame frameCategorical = new JFrame("Categorical Features vs Target Distribution");
        frameCategorical.setLayout(new GridLayout(2, (int) Math.ceil((double) categoricalFeatures.size() / 2)));
        frameCategorical.setSize(1200, 800);
        frameCategorical.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        for (String feature : categoricalFeatures) {
            Dataset<Row> featureData = df.groupBy(feature, "Personal Loan")
                    .count()
                    .groupBy(feature)
                    .pivot("Personal Loan")
                    .agg(functions.sum("count"))
                    .na()
                    .fill(0);

            DefaultCategoryDataset dataset = createDatasetFromSpark(featureData, feature);
            JFreeChart chart = createStackedBarChart(feature, dataset);
            ChartPanel chartPanel = new ChartPanel(chart);
            frameCategorical.add(chartPanel);
        }

        frameCategorical.setVisible(true);
    }

    private static DefaultCategoryDataset createDatasetFromSpark(Dataset<Row> sparkDataset, String feature) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        for (Row row : sparkDataset.collectAsList()) {
            String category = String.valueOf(row.get(0));
            double loan0Count = row.getAs(1) != null ? ((Number) row.getAs(1)).doubleValue() : 0.0;
            double loan1Count = row.getAs(2) != null ? ((Number) row.getAs(2)).doubleValue() : 0.0;

            double total = loan0Count + loan1Count;
            double loan0Proportion = total > 0 ? loan0Count / total : 0.0;
            double loan1Proportion = total > 0 ? loan1Count / total : 0.0;

            dataset.addValue(loan0Proportion, "Loan 0", category);
            dataset.addValue(loan1Proportion, "Loan 1", category);
        }

        return dataset;
    }
    private static JFreeChart createStackedBarChart(String feature, DefaultCategoryDataset dataset) {
        JFreeChart chart = ChartFactory.createStackedBarChart(
                "Feature: " + feature + " vs Personal Loan",
                feature,
                "Proportion",
                dataset
        );

        CategoryPlot plot = (CategoryPlot) chart.getPlot();
        StackedBarRenderer renderer = new StackedBarRenderer();
        plot.setRenderer(renderer);

        return chart;
    }

    //Bar Chart numerical
    private static DefaultCategoryDataset createBarChartDataset(Dataset<Row> df, String feature) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        Dataset<Row> avgData = df.groupBy("Personal Loan")
                .agg(functions.avg(feature).alias("average"));

        avgData.collectAsList().forEach(row -> {
            int loanCategory = row.getInt(0);
            double averageValue = row.getDouble(1);
            dataset.addValue(averageValue, "Personal Loan " + loanCategory, feature);
        });

        return dataset;
    }

    private static JFreeChart createBarChart(String feature, DefaultCategoryDataset dataset) {
        return ChartFactory.createBarChart(
                "Bar Chart: " + feature + " vs Personal Loan",
                "Target",
                "Average Value",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
    }

    //KDE numerical
    private static XYSeries createKdeSeries(Dataset<Row> df, String feature, int loanType) {
        XYSeries series = new XYSeries("Loan " + loanType);
        List<Row> loanData = df.filter("`Personal Loan` = " + loanType).select(feature).collectAsList();
        Map<Double, Double> kdeData = computeKDE(loanData);
        kdeData.forEach(series::add);
        return series;
    }

    private static Map<Double, Double> computeKDE(List<Row> data) {
        Map<Double, Double> kde = new TreeMap<>();
        double bandwidth = 1.0;
        double kernelScale = 1.0 / (Math.sqrt(2 * Math.PI) * bandwidth);

        for (Row row : data) {
            double value;
            if (row.get(0) instanceof Integer) {
                value = row.getInt(0);
            } else if (row.get(0) instanceof Double) {
                value = row.getDouble(0);
            } else {
                continue;
            }

            for (double x = value - 5; x <= value + 5; x += 0.1) {
                double kernelValue = kernelScale * Math.exp(-0.5 * Math.pow((x - value) / bandwidth, 2));
                kde.put(x, kde.getOrDefault(x, 0.0) + kernelValue);
            }
        }

        return kde;
    }

    private static JFreeChart createKdeChart(String feature, XYSeriesCollection dataset, String loanType) {
        JFreeChart chart = ChartFactory.createXYLineChart(
                "KDE Plot: " + feature + " (" + loanType + ")",
                feature,
                "Density",
                dataset
        );

        XYPlot plot = (XYPlot) chart.getPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, loanType.equals("Loan 0") ? Color.BLUE : Color.RED);
        plot.setRenderer(renderer);

        NumberAxis yAxis = (NumberAxis) plot.getRangeAxis();
        yAxis.setAutoRangeIncludesZero(false);

        return chart;
    }
}

