import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Random;

import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

/** Loads the saved model settings and evaluates fresh fold-specific models with 10-fold CV. */
public final class Evaluate {
  private static final int FOLDS = 10;
  private static final int RANDOM_STATE = 42;

  private Evaluate() {}

  public static void main(String[] args) throws Exception {
    if (args.length != 3) {
      throw new IllegalArgumentException("Usage: evaluate <input.arff> <trained.model> <results.json>");
    }
    evaluate(args[0], args[1], args[2]);
  }

  private static void evaluate(String arffPath, String modelPath, String resultsPath) throws Exception {
    Object saved = SerializationHelper.read(modelPath);
    if (!(saved instanceof FilteredClassifier)) {
      throw new IllegalArgumentException("Expected a saved Weka FilteredClassifier model: " + modelPath);
    }
    FilteredClassifier savedTemplate = (FilteredClassifier) saved;
    Instances allData = SvmData.readArff(arffPath);
    checkClassCounts(allData);

    Instances randomized = new Instances(allData);
    randomized.randomize(new Random(RANDOM_STATE));
    randomized.stratify(FOLDS);

    int classes = allData.numClasses();
    double[] accuracies = new double[FOLDS];
    double[] precisions = new double[FOLDS];
    double[] recalls = new double[FOLDS];
    double[] maes = new double[FOLDS];
    double[] rmses = new double[FOLDS];
    double[] f1Scores = new double[FOLDS];
    double[] fitSeconds = new double[FOLDS];
    double[] predictSeconds = new double[FOLDS];
    long allStarted = System.nanoTime();
    Path report = Path.of(resultsPath).toAbsolutePath();
    Files.createDirectories(report.getParent());

    for (int fold = 0; fold < FOLDS; fold++) {
      Instances train = randomized.trainCV(FOLDS, fold);
      Instances test = randomized.testCV(FOLDS, fold);

      // The saved classifier supplies hyperparameters. Each fold gets a new, unfit text filter
      // and a new SVM, trained only on that fold's training partition.
      FilteredClassifier classifier = SvmClassifierFactory.create(train, savedTemplate);
      long fitStarted = System.nanoTime();
      classifier.buildClassifier(train);
      fitSeconds[fold] = (System.nanoTime() - fitStarted) / 1_000_000_000.0;

      double[][] foldConfusion = new double[classes][classes];
      long predictStarted = System.nanoTime();
      for (int row = 0; row < test.numInstances(); row++) {
        Instance instance = test.instance(row);
        int actual = (int) instance.classValue();
        int predicted = (int) classifier.classifyInstance(instance);
        foldConfusion[actual][predicted]++;
      }
      predictSeconds[fold] = (System.nanoTime() - predictStarted) / 1_000_000_000.0;
      double[] metrics = scores(foldConfusion);
      accuracies[fold] = metrics[0];
      precisions[fold] = metrics[1];
      recalls[fold] = metrics[2];
      maes[fold] = metrics[3];
      rmses[fold] = metrics[4];
      f1Scores[fold] = metrics[5];

      writeReport(report, arffPath, modelPath, allData, fold + 1, accuracies, precisions,
          recalls, maes, rmses, f1Scores, fitSeconds, predictSeconds,
          (System.nanoTime() - allStarted) / 1_000_000_000.0, false);
      System.out.printf(Locale.ROOT,
          "Fold %d/%d: fit %.1fs, predict %.1fs, accuracy %.4f, precision %.4f, recall %.4f, MAE %.4f, RMSE %.4f, F1 %.4f%n",
          fold + 1, FOLDS, fitSeconds[fold], predictSeconds[fold], accuracies[fold],
          precisions[fold], recalls[fold], maes[fold], rmses[fold], f1Scores[fold]);
      classifier = null;
      train = null;
      test = null;
      System.gc();
    }

    double elapsed = (System.nanoTime() - allStarted) / 1_000_000_000.0;
    writeReport(report, arffPath, modelPath, allData, FOLDS, accuracies, precisions,
        recalls, maes, rmses, f1Scores, fitSeconds, predictSeconds, elapsed, true);
    System.out.printf(Locale.ROOT,
        "Completed 10-fold CV. Accuracy %.4f ± %.4f; precision %.4f ± %.4f; recall %.4f ± %.4f; "
            + "MAE %.4f ± %.4f; RMSE %.4f ± %.4f; F1 %.4f ± %.4f.%n",
        mean(accuracies), std(accuracies), mean(precisions), std(precisions), mean(recalls), std(recalls),
        mean(maes), std(maes), mean(rmses), std(rmses), mean(f1Scores), std(f1Scores));
    System.out.printf(Locale.ROOT, "Mean build %.1fs/fold; mean prediction %.1fs/fold; total %.1f minutes.%n",
        mean(fitSeconds), mean(predictSeconds), elapsed / 60.0);
    System.out.println("Saved CV report: " + report);
  }

  private static void checkClassCounts(Instances data) {
    int[] counts = new int[data.numClasses()];
    for (int i = 0; i < data.numInstances(); i++) counts[(int) data.instance(i).classValue()]++;
    for (int i = 0; i < counts.length; i++) {
      if (counts[i] < FOLDS) {
        throw new IllegalArgumentException("10-fold CV needs at least 10 examples per class; found "
            + data.classAttribute().value(i) + "=" + counts[i]);
      }
    }
  }

  private static void writeReport(Path path, String arffPath, String modelPath, Instances data,
      int completed, double[] accuracy, double[] precision, double[] recall, double[] mae,
      double[] rmse, double[] f1, double[] fit, double[] predict,
      double elapsed, boolean complete) throws Exception {
    StringBuilder out = new StringBuilder("{\n");
    field(out, "status", complete ? "complete" : "in_progress", true);
    field(out, "input_file", arffPath, true);
    field(out, "model_file", modelPath, true);
    out.append("  \"folds_requested\": ").append(FOLDS).append(",\n");
    out.append("  \"folds_completed\": ").append(completed).append(",\n");
    out.append("  \"stratified\": true,\n  \"random_state\": ").append(RANDOM_STATE).append(",\n");
    out.append("  \"model\": \"Weka FilteredClassifier(StringToWordVector + LibLINEAR linear SVM)\",\n");
    out.append("  \"evaluation_method\": \"Loaded saved model hyperparameters; trained a fresh filter and classifier within each fold.\",\n");
    out.append("  \"total_records\": ").append(data.numInstances()).append(",\n");
    out.append("  \"class_counts\": ").append(classCounts(data)).append(",\n");
    out.append("  \"fold_results\": [\n");
    for (int i = 0; i < completed; i++) {
      out.append(String.format(Locale.ROOT,
          "    {\"fold\":%d,\"fit_seconds\":%.6f,\"predict_seconds\":%.6f,\"accuracy\":%.8f,\"precision\":%.8f,\"recall\":%.8f,\"mae\":%.8f,\"rmse\":%.8f,\"f1_score\":%.8f}%s%n",
          i + 1, fit[i], predict[i], accuracy[i], precision[i], recall[i], mae[i], rmse[i], f1[i],
          i + 1 == completed ? "" : ","));
    }
    out.append("  ],\n");
    if (complete) {
      out.append(String.format(Locale.ROOT,
          "  \"metrics_mean\": {\"accuracy\":%.8f,\"precision\":%.8f,\"recall\":%.8f,\"mae\":%.8f,\"rmse\":%.8f,\"f1_score\":%.8f},%n",
          mean(accuracy), mean(precision), mean(recall), mean(mae), mean(rmse), mean(f1)));
      out.append(String.format(Locale.ROOT,
          "  \"metrics_std\": {\"accuracy\":%.8f,\"precision\":%.8f,\"recall\":%.8f,\"mae\":%.8f,\"rmse\":%.8f,\"f1_score\":%.8f},%n",
          std(accuracy), std(precision), std(recall), std(mae), std(rmse), std(f1)));
      out.append(String.format(Locale.ROOT,
          "  \"fit_seconds_mean\":%.6f,%n  \"predict_seconds_mean\":%.6f,%n",
          mean(fit), mean(predict)));
      out.append(String.format(Locale.ROOT,
          "  \"total_fit_seconds\":%.6f,%n  \"total_predict_seconds\":%.6f,%n",
          sum(fit), sum(predict)));
      out.append("  \"remarks\": [\"Each fold fits StringToWordVector vocabulary and IDF only on that fold's training partition.\", \"LibLINEAR uses class-balanced costs calculated from each fold's training labels.\", \"Fit time includes text vectorization and SVM training; prediction time includes test transformation and classification.\"] ,\n");
    }
    out.append("  \"elapsed_seconds\": ").append(elapsed).append("\n}\n");
    Files.writeString(path, out.toString(), StandardCharsets.UTF_8);
  }

  private static String classCounts(Instances data) {
    int[] counts = new int[data.numClasses()];
    for (int i = 0; i < data.numInstances(); i++) counts[(int) data.instance(i).classValue()]++;
    StringBuilder out = new StringBuilder("{");
    for (int i = 0; i < counts.length; i++) {
      if (i > 0) out.append(',');
      out.append(quote(data.classAttribute().value(i))).append(':').append(counts[i]);
    }
    return out.append('}').toString();
  }

  private static double[] scores(double[][] matrix) {
    double correct = 0, total = 0, macroPrecision = 0, macroRecall = 0, macroF1 = 0;
    double absoluteError = 0, squaredError = 0;
    for (int i = 0; i < matrix.length; i++) {
      double tp = matrix[i][i], support = 0, predicted = 0;
      for (int j = 0; j < matrix.length; j++) {
        support += matrix[i][j];
        predicted += matrix[j][i];
      }
      correct += tp;
      total += support;
      double precision = predicted == 0 ? 0 : tp / predicted;
      double recall = support == 0 ? 0 : tp / support;
      double f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
      macroPrecision += precision;
      macroRecall += recall;
      macroF1 += f1;
      for (int j = 0; j < matrix.length; j++) {
        double error = Math.abs(i - j);
        absoluteError += matrix[i][j] * error;
        squaredError += matrix[i][j] * error * error;
      }
    }
    return new double[] {total == 0 ? 0 : correct / total, macroPrecision / matrix.length,
        macroRecall / matrix.length, total == 0 ? 0 : absoluteError / total,
        total == 0 ? 0 : Math.sqrt(squaredError / total), macroF1 / matrix.length};
  }

  private static double mean(double[] values) {
    double result = 0;
    for (double value : values) result += value;
    return result / values.length;
  }

  private static double std(double[] values) {
    if (values.length < 2) return 0;
    double average = mean(values), sum = 0;
    for (double value : values) sum += (value - average) * (value - average);
    return Math.sqrt(sum / (values.length - 1));
  }

  private static double sum(double[] values) {
    double result = 0;
    for (double value : values) result += value;
    return result;
  }

  private static void field(StringBuilder json, String name, String value, boolean comma) {
    json.append("  ").append(quote(name)).append(": ").append(quote(value)).append(comma ? ",\n" : "\n");
  }

  private static String quote(String value) {
    return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
  }
}
