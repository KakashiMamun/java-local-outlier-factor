/**
 * Copyright 2015-2017 Knowm Inc. (http://knowm.org) and contributors.
 * Copyright 2011-2015 Xeiam LLC (http://xeiam.com) and contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.kakashi.mamun.xchart.charts.scatter;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.XYSeries.XYSeriesRenderStyle;
import org.knowm.xchart.style.Styler;
import org.knowm.xchart.style.markers.SeriesMarkers;

/**
 * Error Bars
 * <p>
 * Demonstrates the following:
 * <ul>
 * <li>Error Bars
 * <li>Using ChartBuilder to Make a Chart
 * <li>List<Number> data sets
 * <li>Setting Series Marker and Marker Color
 * <li>Using a custom decimal pattern
 */
public class ScatterChart implements ExampleChart<XYChart> {

    public static void main(String[] args) {

        ExampleChart<XYChart> exampleChart = new ScatterChart();
        XYChart chart = exampleChart.getChart();
        new SwingWrapper<XYChart>(chart).displayChart();
    }


    public void display(String title, List<Double> xData, List<Double> yData){
        // Create Chart
        XYChart chart = new XYChartBuilder().width(800).height(600).build();

        // Customize Chart
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeriesRenderStyle.Scatter);
        chart.getStyler().setChartTitleVisible(false);
        chart.getStyler().setLegendVisible(false);
        chart.getStyler().setMarkerSize(16);
        chart.getStyler().setYAxisGroupPosition(0, Styler.YAxisPosition.Left);
        chart.setTitle(title);
        // Series
        XYSeries series = chart.addSeries(title, xData, yData);
        series.setMarker(SeriesMarkers.CIRCLE);

        new SwingWrapper<XYChart>(chart).displayChart().setName(title);
    }


    @Override
    public XYChart getChart() {

        // Create Chart
        XYChart chart = new XYChartBuilder().width(800).height(600).build();

        // Customize Chart
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeriesRenderStyle.Scatter);
        chart.getStyler().setChartTitleVisible(false);
        chart.getStyler().setLegendVisible(false);
        chart.getStyler().setMarkerSize(16);
        chart.getStyler().setYAxisGroupPosition(0, Styler.YAxisPosition.Right);
        // Series
        List<Double> xData = new LinkedList<Double>();
        List<Double> yData = new LinkedList<Double>();
        Random random = new Random();
        int size = 1000;
        for (int i = 0; i < size; i++) {
            xData.add(random.nextGaussian() / 1000);
            yData.add(-1000000 + random.nextGaussian());
        }
        XYSeries series = chart.addSeries("Gaussian Blob", xData, yData);
        series.setMarker(SeriesMarkers.CIRCLE);

        return chart;
    }
}