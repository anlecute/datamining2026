import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.meta.FilteredClassifier;
import weka.filters.unsupervised.attribute.StringToWordVector;

/** Creates fresh, untrained components; the dictionary must be learned per fold. */
public final class MnbFactory {
  private MnbFactory() {}

  public static FilteredClassifier create() {
    StringToWordVector filter = new StringToWordVector();
    filter.setLowerCaseTokens(true);
    filter.setOutputWordCounts(true);
    filter.setWordsToKeep(20000);
    filter.setMinTermFreq(2);
    filter.setDoNotOperateOnPerClassBasis(true);
    FilteredClassifier result = new FilteredClassifier();
    result.setFilter(filter);
    result.setClassifier(new NaiveBayesMultinomial());
    return result;
  }

  public static FilteredClassifier copyConfiguration(FilteredClassifier saved) throws Exception {
    if (!(saved.getFilter() instanceof StringToWordVector)
        || !(saved.getClassifier() instanceof NaiveBayesMultinomial)) {
      throw new IllegalArgumentException("Model must contain StringToWordVector and NaiveBayesMultinomial.");
    }
    StringToWordVector filter = new StringToWordVector();
    filter.setOptions(((StringToWordVector) saved.getFilter()).getOptions());
    NaiveBayesMultinomial classifier = new NaiveBayesMultinomial();
    classifier.setOptions(((NaiveBayesMultinomial) saved.getClassifier()).getOptions());
    FilteredClassifier fresh = new FilteredClassifier();
    fresh.setOptions(saved.getOptions());
    fresh.setFilter(filter);
    fresh.setClassifier(classifier);
    return fresh;
  }
}
