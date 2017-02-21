package io.regularization.rl.bandits;

import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.Styler;

/**
 * Created by ptah on 08/02/2017.
 */
public class ChartBuilder {
    public static XYChart buildChart() {
        // Create Chart
        XYChart chart = new XYChartBuilder().width(800).height(600).title("Epsilon comparison").xAxisTitle("Iteration").yAxisTitle("Value").build();

        // Customize Chart
        chart.getStyler().setChartTitleVisible(true);
        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);
        chart.getStyler().setXAxisLogarithmic(true);
        chart.getStyler().setXAxisLabelRotation(45);
        chart.getStyler().setMarkerSize(3);
        chart.getStyler().setPlotTicksMarksVisible(false);

        // chart.getStyler().setXAxisLabelAlignment(TextAlignment.Right);
        // chart.getStyler().setXAxisLabelRotation(90);
        // chart.getStyler().setXAxisLabelRotation(0);

        return chart;
    }
}
