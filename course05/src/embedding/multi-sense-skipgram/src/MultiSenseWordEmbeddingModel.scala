import cc.factorie.model.{Parameters, Weights}
import cc.factorie.optimize.{Trainer, AdaGradRDA}
import cc.factorie.la.DenseTensor1
import cc.factorie.util.Threading
import java.io.{File, PrintWriter}
import java.util.zip.GZIPInputStream
import java.io.FileInputStream
import java.io.OutputStreamWriter
import java.util.zip.GZIPOutputStream
import java.io.BufferedOutputStream
import java.io.FileOutputStream

abstract class MultiSenseWordEmbeddingModel(val opts: EmbeddingOpts) extends Parameters {

  // Algorithm related
  val D                             = opts.dimension.value           // default value is 200
  val S                             = opts.sense.value               // default value is 5
  var V                             = 0                              // vocab size. Will computed in buildVocab() section
  protected val minCount            = opts.minCount.value            // default value is 5
  protected val ignoreStopWords     = opts.ignoreStopWords.value     // default value is 0
  protected val vocabHashSize       = opts.vocabHashSize.value       // default value is 20 M
  protected val maxVocabSize        = opts.vocabSize.value           // default value is 1M
  protected val samplingTableSize   = opts.samplingTableSize.value   // default value is 100 M
  val updateGlobal                  = if (opts.bootVectors.value.size == 0) 1 else opts.updateGlobal.value.toInt
  // Optimization Related
  protected val threads             = opts.threads.value             // default value is 12
  protected val adaGradDelta        = opts.delta.value               // default value is 0.1
  protected val adaGradRate         = opts.rate.value                // default value is 0.025
  // IO Related
  val corpus                        = opts.corpus.value              //
  val storeInBinary                 = opts.binary.value              // default is true
  val loadVocabFilename             = opts.loadVocabFile.value
  val saveVocabFilename             = opts.saveVocabFile.value
  protected val outputFile          = opts.output.value
  private val encoding              = opts.encoding.value            // default is UTF-8
  // Sense/Cluster related
  protected val createClusterlambda = opts.createClusterLambda.value // when to create a new cluster
  val dpmeans                       = if (opts.model.value.equals("NP-MSSG"))          1 else 0
  val kmeans                        = if (opts.model.value.equals("MSSG-KMeans"))      1 else 0
  val maxout                        = if (opts.model.value.equals("MSSG-MaxOut"))      1 else 0
  protected var multiVocabFile      = opts.loadMultiSenseVocabFile.value

  // embedding data structures
  protected var vocab: VocabBuilder           = null
  protected var trainer: HogWildTrainer       = null
  protected var optimizer: AdaGradRDA         = null
  private var corpusLineItr: Iterator[String] = null
  private var train_words: Long               = 0

  // embeddings - global_weights contain the global embeddings(context) and sense_weights contain the embeddings for every word
  // w.r.t to that sense
  var sense_weights: Seq[Seq[Weights]]        = null
  var global_weights: Seq[Weights]            = null

  // clustering related data structures
  protected var clusterCenter: Array[Array[DenseTensor1]] = null   												//每个单词不同sense的聚类中心
  protected var clusterCount: Array[Array[Int]]           = null 																		//每个单词不同sense的更新次数
  protected var clusterActive: Array[Array[Int]] = null	                                                //每个单词不同sense是不是开启
  protected var ncluster: Array[Int] = null // holds the information for # of cluster //每个单词不同的聚类中心数
  protected var countSenses: Array[Int] = null                                                           //每个单词不同sense的数目
  protected var learnMultiVec: Array[Boolean] = null																								//单词是不是要使用MultiSense

  // Component-1
  def buildVocab(minFreq: Int = 5): Unit = {
    vocab = new VocabBuilder(vocabHashSize, samplingTableSize, 0.7) // 0.7 is the load factor
    println("Building Vocab")
    if (loadVocabFilename.size == 0) {
      val corpusLineItr = corpus.endsWith(".gz") match {
        case true  => io.Source.fromInputStream(new GZIPInputStream(new FileInputStream(corpus)), encoding).getLines
        case false => io.Source.fromInputStream(new FileInputStream(corpus), encoding).getLines
      }
      while (corpusLineItr.hasNext) {
       val line = corpusLineItr.next
       line.stripLineEnd.split(' ').foreach(word => vocab.addWordToVocab(word.toLowerCase()))
      }
    }
    else vocab.loadVocab(loadVocabFilename, encoding)

    vocab.sortVocab(minCount, ignoreStopWords, maxVocabSize)  // removes words whose count is less than minCount and sorts by frequency
    vocab.buildSamplingTable() // for getting random word from vocab in O(1) otherwise would O(log |V|)
    vocab.buildSubSamplingTable(opts.sample.value)  // pre-compute subsampling table
    V = vocab.size()
    train_words = vocab.trainWords()
    println("Corpus Stat - Vocab Size :" + V + " Total words (effective) in corpus : " + train_words)
    // save the vocab if the user provides the filename save-vocab
    if (saveVocabFilename.size != 0) {
      println("Saving Vocab into " + saveVocabFilename)
      vocab.saveVocab(saveVocabFilename, storeInBinary, encoding) // for every word, <word><space><count><newline>
      println("Done Saving Vocab")
    }

    learnMultiVec = Array.fill[Boolean](V)(false)
    // load a specific vocab-file for learning multiple-embeddings if learnTopV is 0
    if (multiVocabFile.size != 0) {
        println("Learning multiple embeddings only for those words in " + multiVocabFile)
        for (line <- io.Source.fromFile(multiVocabFile).getLines()) {
          val wrd = line.stripLineEnd
          val id  = vocab.getId(wrd)
          assert(id != -1)
          learnMultiVec(id) = true
        }
        println("Done Loading the socher-multi-vocab-file")
     } else {
        val learnTopV = math.min(opts.learnOnlyTop.value, V)
        println("learn-top-v " + learnTopV)
        println("Learning multiple embeddings for the top most frequent " + learnTopV + " words. ")
        for (v <- 0 until learnTopV)
          learnMultiVec(v) = true
     }
  }

  // Component-2
  def learnEmbeddings(): Unit = {
    println("Learning Embeddings")

    clusterCount  = Array.ofDim[Int](V, S)
    clusterCenter = Array.ofDim[DenseTensor1](V, S)
    ncluster = Array.ofDim[Int](V)
    for (v <- 0 until V) {
      if (dpmeans == 1) {
        ncluster(v) = 1
      } else {
        ncluster(v) = if (learnMultiVec(v)) S else 1
      }
      for (s <- 0 until S) {
        clusterCount(v)(s)  = 1
        clusterCenter(v)(s) = TensorUtils.setToRandom1(new DenseTensor1(D, 0))
      }
    }
    optimizer     = new AdaGradRDA(delta = adaGradDelta, rate = adaGradRate)
    // initialized to random (same manner as in word2vec)
    sense_weights = (0 until V).map(v => (0 until ncluster(v)).map(s => Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0)))))
    if (!opts.bootVectors.value.equals(""))
      load_weights()
    else
      global_weights = (0 until V).map(i => Weights(TensorUtils.setToRandom1(new DenseTensor1(D, 0))))
    optimizer.initializeWeights(this.parameters)
    trainer = new HogWildTrainer(weightsSet = this.parameters, optimizer = optimizer, nThreads = threads, maxIterations = Int.MaxValue)

    println("Initialized Parameters: ")
    println(println("Total memory available to JVM (bytes): " +
        Runtime.getRuntime().totalMemory()))

    val files = (0 until threads).map(i => i)
    Threading.parForeach(files, threads)(workerThread(_))
    println("Done learning embeddings. ")
  }

  def load_weights():Unit = {
   val bootEmbeddingsFile = opts.bootVectors.value
   val bootLineItr = opts.bootVectors.value.endsWith(".gz") match {
      case false => io.Source.fromFile(bootEmbeddingsFile, encoding).getLines
      case true => io.Source.fromInputStream(new GZIPInputStream(new FileInputStream(bootEmbeddingsFile)), encoding).getLines
    }
    val details = bootLineItr.next.stripLineEnd.split(' ').map(_.toInt)
    val Vcheck  = details(0)
    val Dcheck  = details(1)
    assert(Vcheck == V && Dcheck == D)
    println("Bootstrappng vectors : # words : %d , # size : %d".format(V, D))

    val vectors = new Array[DenseTensor1](V)
    for (v <- 0 until V) {
      val line = bootLineItr.next.stripLineEnd.split(' ')
      assert(line.size == 1)
      val word = line(0)
      val org_word = vocab.getWord(v)
      assert(word.equals(org_word) )
      val fields = bootLineItr.next.stripLineEnd.split("\\s+")
        vectors(v) = new DenseTensor1(fields.map(_.toDouble))
     }
     global_weights = (0 until V).map(i => Weights(vectors(i))) // initialized using wordvec random
  }

  // Component-3
  def store(): Unit = {
     println("Now, storing into output... ")
     val out = storeInBinary match {
      case 0 => new java.io.PrintWriter(outputFile, encoding)
      case 1 => new OutputStreamWriter(new GZIPOutputStream(new BufferedOutputStream(new FileOutputStream(outputFile))), encoding)
    }
     // 输出词典大小，维度大小，sense个数，maxout
     out.write("%d %d %d %d\n".format(V, D, S, maxout))
     out.flush()
     // 对词典中的每一个单词，找出每个单词的聚类中心数目
     for (v <- 0 until V) {
       val C = if (learnMultiVec(v)) ncluster(v) else 1
       out.write(vocab.getWord(v) + " " + C)
       //输出每个聚类中心更新的次数
       for (s <- 0 until C) out.write(" " + clusterCount(v)(s))
       out.write("\n"); out.flush();

       //输出global vector
       /*
       val global_embedding = global_weights(v).value
       for (d <- 0 until D) {
         out.write(global_embedding(d) + " ")
       }
       out.write("\n"); out.flush()
       */

       for (s <- 0 until C) {
      	 //输出每一个sense vector
         val sense_embedding = sense_weights(v)(s).value
         for (d <- 0 until D) {
           out.write(sense_embedding(d)+" ")
         }
         //输出每一个sense 的cluster 的context
         var cluster_embedding = clusterCenter(v)(s)
         for(d <- 0 until D){
        	 out.write(cluster_embedding(d)+" ")
         }
         out.write("\n"); out.flush()

        // if maxout method is not used
        if (maxout == 0) {
          val mu = clusterCenter(v)(s) / (1.0 * clusterCount(v)(s))
          for (d <- 0 until D) {
            out.write(mu(d) + " ")
          }
          out.write("\n"); out.flush()
        }
      }
     }
     out.close()
     println("Done, Storing")
  }

  protected def workerThread(id: Int): Unit = {
    val fileLen = new File(corpus).length
    val skipBytes: Long = fileLen / threads * id // skip bytes. skipped bytes is done by other workers
    val lineItr = new FastLineReader(corpus, skipBytes)
    var word_count: Long = 0
    var work = true
    var ndoc = 0
    // worker amount
    val total_words_per_thread = train_words / threads
    while (lineItr.hasNext && work) {
      word_count += process(lineItr.next)
      ndoc += 1
      // print the progress after processing 50 docs (or lines) for Thread-1
      // Approximately reflects the progress for the entire system as all the threads are scheduled "fairly"
      if (id == 1 && ndoc % 50 == 0) {
        val progress = math.min( (word_count/total_words_per_thread.toFloat) * 100, 100.0)
        println(f"Progress: $progress%2.2f" + " %")
      }
      // Once, word_count reaches this limit, ask worker to end
      if (word_count > total_words_per_thread) work = false
    }
  }

  // Override this function in your Embedding Model like SkipGramEmbedding or CBOWEmbedding
  protected def process(doc: String): Int

}
