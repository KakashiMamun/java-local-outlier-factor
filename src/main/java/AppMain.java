import com.github.chen0040.data.evaluators.BinaryClassifierEvaluator;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataQuery;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.frame.Sampler;
import com.github.kakashi.mamun.CBLOF;
import com.github.kakashi.mamun.LDOF;
import com.github.kakashi.mamun.LOCI;
import com.github.kakashi.mamun.LOF;
import lombok.extern.slf4j.Slf4j;
import com.github.kakashi.mamun.xchart.charts.scatter.ScatterChart;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by Kakas on 11/22/2017.
 */
@Slf4j
public class AppMain {

    public static Random random = new Random();

    public static double rand(int min, int max){
        return random.nextInt((max - min) + 1) + min;
    }
    public static void main(String[] args){

        DataQuery.DataFrameQueryBuilder schema = DataQuery.blank()
                .newInput("c1")
                .newInput("c2")
                .newOutput("anomaly")
                .end();

        Sampler.DataSampleBuilder negativeSampler = new Sampler()
                .forColumn("c1").generate((name, index) -> random.nextGaussian() * 0.3 + (index % 2 == 0 ? -2 : 2))
                .forColumn("c2").generate((name, index) -> random.nextGaussian() * 0.3 + (index % 2 == 0 ? -2 : 2))
                .forColumn("anomaly").generate((name, index) -> 0.0)
                .end();

        Sampler.DataSampleBuilder positiveSampler = new Sampler()
                .forColumn("c1").generate((name, index) -> rand(-4, 4))
                .forColumn("c2").generate((name, index) -> rand(-4, 4))
                .forColumn("anomaly").generate((name, index) -> 1.0)
                .end();

        DataFrame data = schema.build();

        data = negativeSampler.sample(data, 10);
        data = positiveSampler.sample(data, 10);


        List<Double> c1 = new ArrayList<>();
        List<Double> c2 = new ArrayList<>();

        for(DataRow dataRow: data.rows()){
            c1.add(dataRow.getCell("c1"));
            c2.add(dataRow.getCell("c2"));
        }

        new ScatterChart().display("Data", c1,c2);

        System.out.println(data.head(10));

        LOF lof = new LOF();
        lof.setParallel(true);
        lof.setMinPtsLB(3);
        lof.setMinPtsUB(10);
        lof.setThreshold(0.5);
        DataFrame learnedData = lof.fitAndTransform(data);

        BinaryClassifierEvaluator evaluator = new BinaryClassifierEvaluator();

        List<Double> lof1 = new ArrayList<>();
        List<Double> lof2 = new ArrayList<>();

        for(int i = 0; i < learnedData.rowCount(); ++i){
            boolean predicted = learnedData.row(i).categoricalTarget().equals("1");
            boolean actual = data.row(i).target() == 1.0;
            evaluator.evaluate(actual, predicted);

            if(predicted){
                lof1.add(data.row(i).getCell("c1"));
                lof2.add(data.row(i).getCell("c2"));
            }

            log.info("predicted: {}\texpected: {}", predicted, actual);
        }


        LDOF ldof = new LDOF();
        learnedData = ldof.fitAndTransform(data);

        evaluator = new BinaryClassifierEvaluator();

        List<Double> ldof1 = new ArrayList<>();
        List<Double> ldof2 = new ArrayList<>();

        for(int i = 0; i < learnedData.rowCount(); ++i) {
            boolean predicted = learnedData.row(i).categoricalTarget().equals("1");
            boolean actual = data.row(i).target() == 1.0;

            evaluator.evaluate(actual, predicted);
            log.info("predicted: {}\texpected: {}", predicted, actual);

            if(predicted){
                ldof1.add(data.row(i).getCell("c1"));
                ldof2.add(data.row(i).getCell("c2"));
            }

        }


        LOCI loci = new LOCI();
        loci.setAlpha(0.5);
        loci.setKSigma(3);
        learnedData = loci.fitAndTransform(data);

        evaluator = new BinaryClassifierEvaluator();

        List<Double> loci1 = new ArrayList<>();
        List<Double> loci2 = new ArrayList<>();

        for(int i = 0; i < learnedData.rowCount(); ++i){
            boolean predicted = learnedData.row(i).categoricalTarget().equals("1");
            boolean actual = data.row(i).target() == 1.0;
            evaluator.evaluate(actual, predicted);
            log.info("predicted: {}\texpected: {}", predicted, actual);

            if(predicted){
                loci1.add(data.row(i).getCell("c1"));
                loci2.add(data.row(i).getCell("c2"));
            }
        }

        CBLOF method = new CBLOF();
        method.setParallel(false);
        learnedData = method.fitAndTransform(data);

        evaluator = new BinaryClassifierEvaluator();

        List<Double> cblof1 = new ArrayList<>();
        List<Double> cblof2 = new ArrayList<>();
        for(int i = 0; i < learnedData.rowCount(); ++i){
            boolean predicted = learnedData.row(i).categoricalTarget().equals("1");
            boolean actual = data.row(i).target() == 1.0;
            evaluator.evaluate(actual, predicted);
            log.info("predicted: {}\texpected: {}", predicted, actual);

            if(predicted){
                cblof1.add(data.row(i).getCell("c1"));
                cblof2.add(data.row(i).getCell("c2"));
            }
        }

        evaluator.report();

//        new ScatterChart().display("LOF", lof1,lof2);
        new ScatterChart().display("LDOF",ldof1,ldof2);
//        new ScatterChart().display("LOCI",loci1,loci2);
//        new ScatterChart().display("CBLOF",cblof1,cblof2);
    }
}
