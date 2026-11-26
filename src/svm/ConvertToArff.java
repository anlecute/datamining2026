/** Converts the preprocessed JSONL dataset into Weka's ARFF format. */
public final class ConvertToArff {
  private ConvertToArff() {}

  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      throw new IllegalArgumentException("Usage: convert <input.jsonl> <output.arff>");
    }
    SvmData.convertJsonlToArff(args[0], args[1]);
  }
}
