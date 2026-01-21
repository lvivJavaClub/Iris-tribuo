package ua.lviv.javaclub.tribuo.demo;

import org.tribuo.DataSource;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.dtree.CARTClassificationTrainer;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.TrainTestSplitter;

import java.io.IOException;
import java.nio.file.Paths;

public class Iris {

    public static void main(String[] args) throws IOException {

        // Load labelled iris data
        var irisHeaders = new String[]{"sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"};
        DataSource<Label> irisData =
                new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get("data/bezdekIris.data"),
                        /* Output column   */ irisHeaders[4],
                        /* Column headers  */ irisHeaders);

        // Split iris data into training set (70%) and test set (30%)
        var splitIrisData = new TrainTestSplitter<>(irisData,
                /* Train fraction */ 0.7,
                /* RNG seed */ 1L);

        // CARTClassificationTrainer -----------------------------------------------------------------------------------
        var trainData = new MutableDataset<>(splitIrisData.getTrain());
        var testData = new MutableDataset<>(splitIrisData.getTest());

        var example = testData.getExample(0);
        System.out.print("example:");
        System.out.println(example);

        // We can train a decision tree
        var cartTrainer = new CARTClassificationTrainer();
        Model<Label> tree = cartTrainer.train(trainData);
        // Each prediction is a map from the output names (i.e. the labels) to the scores/probabilities

        Prediction<Label> prediction1 = tree.predict(example);
        System.out.println("CARTClassificationTrainer:");
        System.out.println(prediction1);
        System.out.print("CARTClassificationTrainer result: ");
        System.out.println(prediction1.getOutput().getLabel());

        LabelEvaluation evaluation1 = new LabelEvaluator().evaluate(tree, testData);
        // we can inspect the evaluation manually
        double acc1 = evaluation1.accuracy();
        System.out.print("tree accuracy:");
        System.out.println(acc1);

        // LogisticRegressionTrainer -----------------------------------------------------------------------------------
        var linearTrainer = new LogisticRegressionTrainer();
        Model<Label> linear = linearTrainer.train(trainData);

        // Finally we make predictions on unseen data
        // Each prediction is a map from the output names (i.e. the labels) to the scores/probabilities
        Prediction<Label> prediction2 = linear.predict(testData.getExample(0));
        System.out.println("LogisticRegressionTrainer:");
        System.out.println(prediction2);
        System.out.print("LogisticRegressionTrainer result: ");
        System.out.println(prediction2.getOutput().getLabel());

        // Or we can evaluate the full test dataset, calculating the accuracy
        LabelEvaluation evaluation2 = new LabelEvaluator().evaluate(linear, testData);
        // we can inspect the evaluation manually
        double acc2 = evaluation2.accuracy();
        System.out.print("linear accuracy:");
        System.out.println(acc2);

        // or print a formatted evaluation string
        System.out.println(evaluation2);
    }
}