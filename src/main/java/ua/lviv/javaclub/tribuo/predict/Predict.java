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

//        ArrayExample(numFeatures=4,output=Iris-setosa,weight=1.0,
//        features=[ (petalLength, 1.3), (petalWidth, 0.3), (sepalLength, 4.5), (sepalWidth, 2.3), ])
        var example = new ArrayExample<>(
                new Label("UNKNOWN") // dummy label, ignored in prediction
        );
        example.add(new Feature("sepalLength", 4.5));
        example.add(new Feature("sepalWidth", 2.3));
        example.add(new Feature("petalLength", 1.3));
        example.add(new Feature("petalWidth", 0.3));

        var prediction = model.predict(example);
        System.out.println("Predicted label: " + prediction.getOutput().getLabel());
    }
}