package ua.lviv.javaclub.tribuo.predict;

import org.tribuo.Feature;
import org.tribuo.Model;
import org.tribuo.classification.Label;
import org.tribuo.impl.ArrayExample;

import java.io.IOException;
import java.nio.file.Path;

public class Predict {

    public static void main(String[] args) throws IOException {

        Path modelPath = Path.of("models/model.tribuo");

        @SuppressWarnings("unchecked")
        Model<Label> model = (Model<Label>) Model.deserializeFromFile(
                modelPath
        );

        var example = new ArrayExample<>(
                new Label("UNKNOWN") // dummy label, ignored in prediction
        );

//        ArrayExample(numFeatures=6,output=1,weight=1.0,
//        features=[(amount, 406864.17)(newbalanceDest, 0.0), (newbalanceOrig, 0.0), (oldbalanceDest, 0.0), (oldbalanceOrg, 406864.17), (step, 571.0), ])

        example.add(new Feature("step", 571.0));
        example.add(new Feature("amount", 406864.17));
        example.add(new Feature("oldbalanceOrg", 406864.17));
        example.add(new Feature("newbalanceOrig", 0.0));
        example.add(new Feature("oldbalanceDest", 0.0));
        example.add(new Feature("newbalanceDest", 0.0));

        var prediction = model.predict(example);
        System.out.println("Predicted label: " + prediction.getOutput().getLabel());
    }
}