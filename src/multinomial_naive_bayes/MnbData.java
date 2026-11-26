import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/** Shared data loading and JSONL-to-ARFF conversion helpers. */
public final class MnbData {
  public static final String[] LABELS = {"Negative", "Neutral", "Positive"};

  private MnbData() {}

  public static Instances readArff(String path) throws Exception {
    Instances data = DataSource.read(path);
    if (data == null || data.numInstances() == 0) {
      throw new IllegalArgumentException("ARFF has no instances: " + path);
    }
    data.setClassIndex(data.numAttributes() - 1);
    if (!data.classAttribute().isNominal()) {
      throw new IllegalArgumentException("Last ARFF attribute must be nominal class.");
    }
    if (data.numAttributes() != 2 || !data.attribute(0).name().equals("review_text")
        || !data.attribute(0).isString() || !data.classAttribute().name().equals("sentiment_class")
        || data.numClasses() != LABELS.length) {
      throw new IllegalArgumentException("Expected review_text STRING and sentiment_class {Negative,Neutral,Positive}.");
    }
    for (int i = 0; i < LABELS.length; i++) {
      if (!LABELS[i].equals(data.classAttribute().value(i)))
        throw new IllegalArgumentException("Class order must be Negative, Neutral, Positive.");
    }
    for (int i = 0; i < data.numInstances(); i++) {
      if (data.instance(i).classIsMissing() || data.instance(i).isMissing(0)
          || data.instance(i).stringValue(0).isBlank())
        throw new IllegalArgumentException("Missing class or review text in ARFF instance " + (i + 1));
    }
    return data;
  }

  public static void convertJsonlToArff(String jsonlPath, String arffPath) throws Exception {
    Path input = Path.of(jsonlPath);
    Path output = Path.of(arffPath).toAbsolutePath();
    if (!Files.isRegularFile(input)) throw new IllegalArgumentException("JSONL input not found: " + input);
    Files.createDirectories(output.getParent());
    Map<String, Integer> counts = new TreeMap<>();
    int records = 0;
    try (BufferedReader source = Files.newBufferedReader(input, StandardCharsets.UTF_8);
         BufferedWriter target = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
      target.write("@RELATION amazon_reviews_sentiment\n\n");
      target.write("@ATTRIBUTE review_text STRING\n");
      target.write("@ATTRIBUTE sentiment_class {Negative,Neutral,Positive}\n\n");
      target.write("@DATA\n");
      String line;
      while ((line = source.readLine()) != null) {
        records++;
        Map<String, String> row;
        try {
          row = new JsonObjectReader(line).readStringFields();
        } catch (Exception error) {
          throw new IllegalArgumentException("Invalid JSON at line " + records, error);
        }
        String review = row.get("review_text");
        if (review == null) {
          String title = row.get("title");
          String body = row.get("text");
          review = ((title == null ? "" : title.trim()) + " " + (body == null ? "" : body.trim())).trim();
        }
        String label = row.get("sentiment_class");
        if (review.isBlank() || !java.util.Arrays.asList(LABELS).contains(label)) {
          throw new IllegalArgumentException("Invalid review text or sentiment_class at JSONL line " + records);
        }
        target.write(arffQuote(review) + "," + label + "\n");
        counts.merge(label, 1, Integer::sum);
      }
    }
    if (records == 0) throw new IllegalArgumentException("JSONL input is empty: " + input);
    System.out.printf(Locale.ROOT, "Converted %,d JSONL records to ARFF: %s%n", records, output);
    System.out.println("Class counts: " + counts);
  }

  private static String arffQuote(String value) {
    StringBuilder normalized = new StringBuilder(value.length());
    boolean pendingSpace = false;
    for (int offset = 0; offset < value.length();) {
      int codePoint = value.codePointAt(offset);
      offset += Character.charCount(codePoint);
      if (Character.isWhitespace(codePoint) || Character.isSpaceChar(codePoint)) {
        pendingSpace = normalized.length() > 0;
      } else {
        if (pendingSpace) normalized.append(' ');
        normalized.appendCodePoint(codePoint);
        pendingSpace = false;
      }
    }
    return "'" + normalized.toString().replace("\\", "\\\\").replace("'", "\\'") + "'";
  }

  /** Minimal JSON object reader for the project's flat JSONL records. */
  private static final class JsonObjectReader {
    private final String source;
    private int position;

    JsonObjectReader(String source) { this.source = source; }

    Map<String, String> readStringFields() {
      Map<String, String> fields = new TreeMap<>();
      skipWhitespace();
      expect('{');
      skipWhitespace();
      if (take('}')) return fields;
      while (true) {
        skipWhitespace();
        String key = readString();
        skipWhitespace();
        expect(':');
        skipWhitespace();
        if (peek() == '"') fields.put(key, readString());
        else skipValue();
        skipWhitespace();
        if (take('}')) break;
        expect(',');
      }
      skipWhitespace();
      if (position != source.length()) throw error("Unexpected trailing JSON data");
      return fields;
    }

    private String readString() {
      expect('"');
      StringBuilder value = new StringBuilder();
      while (position < source.length()) {
        char c = source.charAt(position++);
        if (c == '"') return value.toString();
        if (c != '\\') {
          if (c < 0x20) throw error("Unescaped control character in JSON string");
          value.append(c);
          continue;
        }
        if (position >= source.length()) throw error("Incomplete JSON escape");
        char escape = source.charAt(position++);
        switch (escape) {
          case '"': value.append('"'); break;
          case '\\': value.append('\\'); break;
          case '/': value.append('/'); break;
          case 'b': value.append('\b'); break;
          case 'f': value.append('\f'); break;
          case 'n': value.append('\n'); break;
          case 'r': value.append('\r'); break;
          case 't': value.append('\t'); break;
          case 'u':
            if (position + 4 > source.length()) throw error("Incomplete Unicode escape");
            try {
              value.append((char) Integer.parseInt(source.substring(position, position + 4), 16));
            } catch (NumberFormatException exception) {
              throw error("Invalid Unicode escape");
            }
            position += 4;
            break;
          default: throw error("Invalid JSON escape at character " + (position - 1));
        }
      }
      throw error("Unterminated JSON string");
    }

    private void skipValue() {
      char c = peek();
      if (c == '"') {
        readString();
      } else if (c == '{' || c == '[') {
        java.util.ArrayDeque<Character> closing = new java.util.ArrayDeque<>();
        closing.push(c == '{' ? '}' : ']');
        position++;
        while (!closing.isEmpty()) {
          if (position >= source.length()) throw error("Unterminated JSON container");
          c = source.charAt(position);
          if (c == '"') readString();
          else if (c == '{' || c == '[') {
            closing.push(c == '{' ? '}' : ']');
            position++;
          } else if (c == '}' || c == ']') {
            if (closing.pop() != c) throw error("Mismatched JSON container");
            position++;
          } else position++;
        }
      } else {
        int start = position;
        while (position < source.length() && source.charAt(position) != ','
            && source.charAt(position) != '}' && source.charAt(position) != ']') position++;
        if (start == position) throw error("Missing JSON value");
      }
    }

    private void skipWhitespace() {
      while (position < source.length() && Character.isWhitespace(source.charAt(position))) position++;
    }
    private char peek() {
      if (position >= source.length()) throw error("Unexpected end of JSON");
      return source.charAt(position);
    }
    private boolean take(char expected) {
      if (position < source.length() && source.charAt(position) == expected) {
        position++;
        return true;
      }
      return false;
    }
    private void expect(char expected) {
      if (!take(expected)) throw error("Expected '" + expected + "'");
    }
    private IllegalArgumentException error(String message) {
      return new IllegalArgumentException(message + " at character " + position);
    }
  }
}


