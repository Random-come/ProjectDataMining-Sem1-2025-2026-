import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

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

            RandomForest model = new RandomForest();
            Instant start = Instant.now();
            model.buildClassifier(train);
            long buildMillis = Duration.between(start, Instant.now()).toMillis();

            Evaluation evaluation = new Evaluation(train);
            evaluation.evaluateModel(model, test);

            SerializationHelper.write(outputPath.toString(), model);

            System.out.printf("Model saved to %s%n", outputPath);
            System.out.printf("Training runtime: %d ms%n", buildMillis);
            System.out.printf("Accuracy on test set: %.2f%%%n", evaluation.pctCorrect());
        } catch (Exception e) {
            System.err.printf("Failed to build RandomForest model: %s%n", e.getMessage());
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


