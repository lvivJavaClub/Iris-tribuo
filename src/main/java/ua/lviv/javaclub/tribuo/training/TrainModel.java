package ua.lviv.javaclub.tribuo.training;

import org.tribuo.DataSource;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.xgboost.XGBoostClassificationTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.TrainTestSplitter;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class TrainModel {
    private static final Set<Integer> COLUMNS_TO_REMOVE =
            Set.of(1, 3, 6, 10);

    public static void cleanup() throws IOException {
        System.out.println("cleanup started");

        Path input = Paths.get("./data/Fraud.csv");
        if (!Files.exists(input)) {
            System.err.println("Error: data/Fraud.csv not found. Please download the dataset as described in README.MD.");
            System.exit(1);
        }
        Path output = Paths.get("data/Fraud_noheader.csv");

        try (BufferedReader reader = Files.newBufferedReader(input);
             BufferedWriter writer = Files.newBufferedWriter(output)) {

            String line;
            boolean isFirstLine = true;

            while ((line = reader.readLine()) != null) {
                if (isFirstLine) {
                    isFirstLine = false;
                    continue;
                }

                String[] cols = line.split(",", -1);
                List<String> filtered = new ArrayList<>();

                for (int i = 0; i < cols.length; i++) {
                    if (!COLUMNS_TO_REMOVE.contains(i)) {
                        filtered.add(cols[i]);
                    }
                }

                writer.write(String.join(",", filtered));
                writer.newLine();
            }
        }
        System.out.println("cleanup finished");
    }

    public static void main(String[] args) throws IOException {

        cleanup();

        var irisHeaders = new String[]{
                "step",
//                "type",
                "amount",
//                "nameOrig",
                "oldbalanceOrg",
                "newbalanceOrig",
//                "nameDest",
                "oldbalanceDest",
                "newbalanceDest",
                "isFraud",
//                "isFlaggedFraud"
        };
        DataSource<Label> irisData =
                new CSVLoader<>(
                        new LabelFactory()
                ).loadDataSource(Paths.get("data/Fraud_noheader.csv"), irisHeaders[6], irisHeaders);

        // Split iris data into training set (70%) and test set (30%)
        var splitIrisData = new TrainTestSplitter<>(irisData, 0.7, 1L);

        var trainData = new MutableDataset<>(splitIrisData.getTrain());
        var testData = new MutableDataset<>(splitIrisData.getTest());

        // LogisticRegressionTrainer OR CARTClassificationTrainer
//        var cartTrainer = new LogisticRegressionTrainer();
//        var cartTrainer = new LibLinearClassificationTrainer(
//                new LinearClassificationType(LinearClassificationType.LinearType.L2R_L2LOSS_SVC_DUAL),
//                1.0,
//                10,
//                1.0
//        );

        var cartTrainer = new XGBoostClassificationTrainer(
                200
        );

        long startNs = System.nanoTime();
        System.out.println("train....");

        Model<Label> model = cartTrainer.train(trainData);

        long endNs = System.nanoTime();
        double durationSeconds = (endNs - startNs) / 1_000_000_000.0;
        System.out.printf("Execution time: %.6f s%n", durationSeconds);

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