import weka.classifiers.Evaluation;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.classifiers.CostMatrix;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;

public class RandomForestModel {

    private static final String MODEL_FILENAME = "RANDOMFOREST.model";

    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println("Usage: java RandomForestModel <train-arff> <test-arff> <output-dir-or-file>");
            System.exit(1);
        }

        try {
            Path trainPath = normalize(args[0]);
            Path testPath = normalize(args[1]);
            Path outputPath = resolveOutputPath(normalize(args[2]), MODEL_FILENAME);

            Instances train = loadInstances(trainPath);
            Instances test = loadInstances(testPath);
            ensureClassIndex(train);
            ensureClassIndex(test);

            // Create cost matrix to penalize False Negatives (FN)
            // Cost matrix format: [cost(0->0), cost(0->1)] 
            //                    [cost(1->0), cost(1->1)]
            // Where cost(1->0) is FN cost (predict 0 when true is 1) - should be high
            //      cost(0->1) is FP cost (predict 1 when true is 0) - can be lower
            CostMatrix costMatrix = new CostMatrix(2);
            costMatrix.setCell(0, 0, 0.0);  // TN: no cost
            costMatrix.setCell(0, 1, 1.0);  // FP: low cost
            costMatrix.setCell(1, 0, 5.0); // FN: high cost (penalize heavily)
            costMatrix.setCell(1, 1, 0.0);  // TP: no cost
            
            // Cost Matrix (to penalize FN):")
            // Predict 0 when True 0 (TN): cost = 0.0")
            // Predict 1 when True 0 (FP): cost = 1.0")
            // Predict 0 when True 1 (FN): cost = 10.0  <-- High penalty")
            // Predict 1 when True 1 (TP): cost = 0.0")

            // Create base RandomForest classifier
            RandomForest baseClassifier = new RandomForest();
            
            // Wrap with CostSensitiveClassifier
            CostSensitiveClassifier model = new CostSensitiveClassifier();
            model.setClassifier(baseClassifier);
            model.setCostMatrix(costMatrix);
            model.setMinimizeExpectedCost(false); // Minimize cost during training

            Instant start = Instant.now();
            model.buildClassifier(train);
            long buildMillis = Duration.between(start, Instant.now()).toMillis();

            // Evaluate on test set
            Evaluation evaluation = new Evaluation(train, costMatrix);
            evaluation.evaluateModel(model, test);

            SerializationHelper.write(outputPath.toString(), model);

            System.out.printf("\nModel saved to %s%n", outputPath);
            System.out.printf("Training runtime: %d ms%n", buildMillis);
            System.out.printf("Accuracy on test set: %.2f%%%n", evaluation.pctCorrect());
            System.out.printf("Total cost: %.2f%n", evaluation.totalCost());
            System.out.println("\nConfusion Matrix:");
            System.out.println(evaluation.toMatrixString());
            System.out.println("\nDetailed Accuracy By Class:");
            System.out.println(evaluation.toClassDetailsString());
        } catch (Exception e) {
            System.err.printf("Failed to build RandomForest model: %s%n", e.getMessage());
            e.printStackTrace();
            System.exit(2);
        }
    }

    private static Path normalize(String input) {
        return Paths.get(input).toAbsolutePath().normalize();
    }

    private static Instances loadInstances(Path path) throws Exception {
        if (!Files.exists(path)) {
            throw new IllegalArgumentException("File does not exist: " + path);
        }
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(path.toString());
        return source.getDataSet();
    }

    private static void ensureClassIndex(Instances instances) {
        if (instances.classIndex() == -1) {
            instances.setClassIndex(instances.numAttributes() - 1);
        }
    }

    private static Path resolveOutputPath(Path candidate, String fileName) throws Exception {
        if (Files.exists(candidate) && Files.isDirectory(candidate)) {
            return candidate.resolve(fileName);
        }
        if (candidate.toString().toLowerCase().endsWith(".model")) {
            Path parent = candidate.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            return candidate;
        }
        Files.createDirectories(candidate);
        return candidate.resolve(fileName);
    }
}


