package ua.lviv.javaclub.tribuo.training;

import org.tribuo.DataSource;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.liblinear.LibLinearClassificationTrainer;
import org.tribuo.classification.liblinear.LinearClassificationType;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.TrainTestSplitter;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class TrainModel {

    public static void main(String[] args) throws IOException {

        var irisHeaders = new String[]{"sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"};
        DataSource<Label> irisData =
                new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get("data/bezdekIris.data"),
                        /* Output column   */ irisHeaders[4],
                        /* Column headers  */ irisHeaders);

        // Split iris data into training set (70%) and test set (30%)
        var splitIrisData = new TrainTestSplitter<>(irisData, 0.7, 1L);

        var trainData = new MutableDataset<>(splitIrisData.getTrain());
        var testData = new MutableDataset<>(splitIrisData.getTest());

        // LogisticRegressionTrainer OR CARTClassificationTrainer
//        var cartTrainer = new LogisticRegressionTrainer();
        var cartTrainer = new LibLinearClassificationTrainer(
                new LinearClassificationType(LinearClassificationType.LinearType.L2R_L2LOSS_SVC_DUAL),
                1.0,
                10,
                1.0
        );
        Model<Label> model = cartTrainer.train(trainData);

        // Evaluate the model on the test set
        LabelEvaluation evaluation = new LabelEvaluator().evaluate(model, testData);
        System.out.println("Model Evaluation:");
        System.out.println(evaluation.toString());
        System.out.println("Accuracy: " + evaluation.accuracy());

        var modelPath = Path.of("models/model.tribuo");
        model.serializeToFile(modelPath);
        System.out.println("Model saved to: " + modelPath.toAbsolutePath());
    }
}