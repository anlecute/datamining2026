import java.io.File;
import java.util.Locale;

import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.SerializationHelper;

/** Trains the final SVM on the complete ARFF dataset and saves it. */
public final class Train {
  private Train() {}

  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      throw new IllegalArgumentException("Usage: train <input.arff> <output.model>");
    }
    Instances data = SvmData.readArff(args[0]);
    FilteredClassifier classifier = SvmClassifierFactory.create(data);
    long started = System.nanoTime();
    classifier.buildClassifier(data);
    double seconds = (System.nanoTime() - started) / 1_000_000_000.0;

    File output = new File(args[1]).getAbsoluteFile();
    File parent = output.getParentFile();
    if (parent != null) parent.mkdirs();
    SerializationHelper.write(output.getPath(), classifier);
    System.out.printf(Locale.ROOT, "Trained SVM on %,d records in %.2f seconds.%n", data.numInstances(), seconds);
    System.out.println("Saved model: " + output);
  }
}
