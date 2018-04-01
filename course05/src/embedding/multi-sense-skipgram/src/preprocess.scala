import cc.factorie.app.nlp.segment.DeterministicTokenizer
import cc.factorie.app.nlp.segment.DeterministicSentenceSegmenter
import cc.factorie.app.nlp.Document
import java.io.PrintWriter
import java.nio.charset.Charset

object preprocess {
  def main(args : Array[String]) {
      println("Default Charset=" + Charset.defaultCharset());
    	println("file.encoding=" + System.getProperty("file.encoding"));
    	
      val inputFile = args(0)
      val outFile = args(1)
      val out = new PrintWriter(outFile, "ISO-8859-15")
      val tokenizer = new DeterministicTokenizer
      val sen_tokenizer = new DeterministicSentenceSegmenter
      for (line <- io.Source.fromFile(inputFile, "ISO-8859-15").getLines) {
            
            val doc = new Document(line)
            tokenizer.process(doc)
            sen_tokenizer.process(doc)
            if (doc.tokens.size > 10) {
              for (token <- doc.tokens)
                out.print(token.string + " ")
               out.print("\n")
               out.flush()
            }
            else {
              println(doc.tokens.toArray)
            }
            
      }
      out.close()
  }
}