import java.util.Locale;

import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.unsupervised.attribute.StringToWordVector;

/** Creates fresh Weka pipelines, including fold-specific text filters and class costs. */
public final class SvmClassifierFactory {
  private static final int DEFAULT_WORDS_TO_KEEP = 50000;
  private static final int DEFAULT_MIN_TERM_FREQUENCY = 2;
  private static final int DEFAULT_NGRAM_MIN = 1;
  private static final int DEFAULT_NGRAM_MAX = 2;
  private static final double DEFAULT_C = 1.0;
  private static final double DEFAULT_EPSILON = 0.01;

  private SvmClassifierFactory() {}

  public static FilteredClassifier create(Instances trainingData) {
    return create(trainingData, null);
  }

  /**
   * Uses the saved model's settings but creates a new filter and SVM for this training partition.
   * This is used by CV so the vocabulary and IDF never come from the full-data trained model.
   */
  public static FilteredClassifier create(Instances trainingData, FilteredClassifier savedTemplate) {
    StringToWordVector filter = new StringToWordVector();
    filter.setAttributeIndices("first");
    filter.setWordsToKeep(savedTemplate == null ? DEFAULT_WORDS_TO_KEEP : savedFilter(savedTemplate).getWordsToKeep());
    filter.setMinTermFreq(savedTemplate == null ? DEFAULT_MIN_TERM_FREQUENCY : savedFilter(savedTemplate).getMinTermFreq());
    filter.setOutputWordCounts(savedTemplate == null || savedFilter(savedTemplate).getOutputWordCounts());
    filter.setTFTransform(savedTemplate == null || savedFilter(savedTemplate).getTFTransform());
    filter.setIDFTransform(savedTemplate == null || savedFilter(savedTemplate).getIDFTransform());
    filter.setLowerCaseTokens(savedTemplate == null || savedFilter(savedTemplate).getLowerCaseTokens());
    filter.setDoNotOperateOnPerClassBasis(savedTemplate == null || savedFilter(savedTemplate).getDoNotOperateOnPerClassBasis());
    filter.setTokenizer(newTokenizer(savedTemplate));

    LibLINEAR svm = new LibLINEAR();
    if (savedTemplate == null) {
      // Solver 2 is primal L2-loss support vector classification.
      svm.setSVMType(new SelectedTag(2, LibLINEAR.TAGS_SVMTYPE));
      svm.setCost(DEFAULT_C);
      svm.setEps(DEFAULT_EPSILON);
    } else {
      LibLINEAR savedSvm = (LibLINEAR) savedTemplate.getClassifier();
      svm.setSVMType(savedSvm.getSVMType());
      svm.setCost(savedSvm.getCost());
      svm.setEps(savedSvm.getEps());
      svm.setNormalize(savedSvm.getNormalize());
    }
    svm.setWeights(balancedClassWeights(trainingData));

    FilteredClassifier result = new FilteredClassifier();
    result.setFilter(filter);
    result.setClassifier(svm);
    return result;
  }

  private static StringToWordVector savedFilter(FilteredClassifier savedTemplate) {
    if (!(savedTemplate.getFilter() instanceof StringToWordVector)) {
      throw new IllegalArgumentException("Saved model does not contain StringToWordVector.");
    }
    return (StringToWordVector) savedTemplate.getFilter();
  }

  private static NGramTokenizer newTokenizer(FilteredClassifier savedTemplate) {
    NGramTokenizer tokenizer = new NGramTokenizer();
    if (savedTemplate == null) {
      tokenizer.setNGramMinSize(DEFAULT_NGRAM_MIN);
      tokenizer.setNGramMaxSize(DEFAULT_NGRAM_MAX);
    } else if (savedFilter(savedTemplate).getTokenizer() instanceof NGramTokenizer) {
      NGramTokenizer saved = (NGramTokenizer) savedFilter(savedTemplate).getTokenizer();
      tokenizer.setNGramMinSize(saved.getNGramMinSize());
      tokenizer.setNGramMaxSize(saved.getNGramMaxSize());
      tokenizer.setDelimiters(saved.getDelimiters());
    } else {
      throw new IllegalArgumentException("Saved model uses an unsupported tokenizer.");
    }
    return tokenizer;
  }

  private static String balancedClassWeights(Instances data) {
    int classes = data.numClasses();
    int[] counts = new int[classes];
    for (int i = 0; i < data.numInstances(); i++) {
      Instance instance = data.instance(i);
      if (!instance.classIsMissing()) counts[(int) instance.classValue()]++;
    }
    StringBuilder weights = new StringBuilder();
    for (int cls = 0; cls < classes; cls++) {
      if (cls > 0) weights.append(' ');
      double weight = counts[cls] == 0 ? 1.0 : (double) data.numInstances() / (classes * counts[cls]);
      weights.append(String.format(Locale.ROOT, "%.12f", weight));
    }
    return weights.toString();
  }
}
