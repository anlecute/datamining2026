import java.io.File;
import java.util.Locale;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.SerializationHelper;

/** Trains and saves a complete filter + classifier on the supplied ARFF. */
public final class train {
  public static void main(String[] args) throws Exception {
    if (args.length != 2) throw new IllegalArgumentException("Usage: train <input.arff> <output.model>");
    Instances data = MnbData.readArff(args[0]);
    FilteredClassifier model = MnbFactory.create();
    long start = System.nanoTime();
    model.buildClassifier(data);
    File output = new File(args[1]).getAbsoluteFile();
    if (!output.getParentFile().isDirectory() && !output.getParentFile().mkdirs())
      throw new IllegalStateException("Cannot create directory: " + output.getParent());
    SerializationHelper.write(output.getPath(), model);
    System.out.printf(Locale.ROOT, "Trained on %,d records in %.2f seconds.%nSaved model: %s%n", data.numInstances(), (System.nanoTime()-start)/1e9, output);
  }
}
