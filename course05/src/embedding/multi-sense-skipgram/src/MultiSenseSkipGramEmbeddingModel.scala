import cc.factorie.util.DoubleAccumulator
import cc.factorie.la.{DenseTensor1, WeightsMapAccumulator}
import cc.factorie.optimize.Example
import scala.collection.mutable

class MultiSenseSkipGramEmbeddingModel(override val opts: EmbeddingOpts) extends MultiSenseWordEmbeddingModel(opts) {
  val negative = opts.negative.value
  val window = opts.window.value
  val rng = new util.Random
  val sample = opts.sample.value.toDouble
  
  override def process(doc: String): Int = {
    // given a document, below line splits by space and converts each word to Int (by vocab.getId) and filters out words not in vocab
    // id of a word is its freq-rank in the corpus
    var sen = doc.stripLineEnd.split(' ').map(word => vocab.getId(word.toLowerCase())).filter(id => id != -1)
    val wordCount = sen.size

    // 
    // subsampling the words : refer to Google's word2vec NIPS paper to understand this
    //
    if (sample > 0)
      sen = sen.filter(id => subSample(id) != -1)

    val senLength = sen.size
    for (senPosition <- 0 until senLength) {
      val currWord = sen(senPosition)
      
      //
      // dynamic window-size as in word2vec. 
      //
      val b =  rng.nextInt(window)
      
      //
      // make the contexts
      //
      val contexts = new mutable.ArrayBuffer[Int]
      for (a <- b until window * 2 + 1 - b) if (a != window) {
        val c = senPosition - window + a
        if (c >= 0 && c < senLength)
          contexts += sen(c)
      }
      
      // predict the sense of the word given the contexts. P(word-sense | word, contexts)
      var rightSense = 0
      if (kmeans == 1)
         rightSense  = cbow_predict_kmeans(currWord, contexts)
      else if (dpmeans == 1)
          rightSense = cbow_predict_dpmeans(currWord, contexts)
      else
          rightSense = cbow_predict(currWord, contexts)
      
      // make the examples. trainer is HogWild!
      contexts.foreach(context => {
        // postive example
        trainer.processExample(new MSCBOWSkipGramNegSamplingExample(this, currWord, rightSense, context, 1))
        // for each POS example, make negative NEG examples. 
        //vocab.getRandWordId would get random word proportional (unigram-prob)^(0.75). 
        (0 until negative).foreach(neg => trainer.processExample(new MSCBOWSkipGramNegSamplingExample(this, currWord, rightSense, vocab.getRandWordId, -1)))
        
      })
    }
    return wordCount
  }
  
  // Predict the sense of the word using the context embeddings (sum of context word embeddings)
  // This one does not need cluster centers at all
  // It is very similar to Jason Weston's MaxMF factorization (Multiple-latent representations per user) and Bengio's   Maxout-Networks
  // Works quite well and gives almost same-performance as k-means and dp-means style sense prediction
  // Major Pros: No need to maintain to cluster centers. So less memory.  |V| X |S| X |D| X 8 bytes saved
  def cbow_predict(word : Int, contexts: Seq[Int]): Int = {
    val contextsEmbedding = new DenseTensor1(D, 0)    
    contexts.foreach(context => contextsEmbedding.+=(global_weights(context).value))
    var sense = 0
    if (learnMultiVec(word)) {
      var maxdot = contextsEmbedding.dot(sense_weights(word)(0).value)
      for (s <- 1 until S) {
        val dot = contextsEmbedding.dot(sense_weights(word)(s).value)
        if (dot > maxdot) {
          maxdot = dot
          sense = s
        }
      }
    }
    sense
  }
  
  // Use the cluster centers to predict the sense. Similar to Kmeans
  def cbow_predict_kmeans(word: Int, contexts: Seq[Int]): Int = {   
      val contextsEmbedding = new DenseTensor1(D, 0)    
      contexts.foreach(context => contextsEmbedding.+=(global_weights(context).value))
      var sense = 0
       
      if (learnMultiVec(word)) {
         var minDist = Double.MaxValue
         for (s <- 0 until ncluster(word)) { 
            val mu = clusterCenter(word)(s)/(clusterCount(word)(s)) 
            val dist = 1 - TensorUtils.cosineDistance(contextsEmbedding, mu) 
            if (dist < minDist) {
              minDist = dist
              sense = s
            }
         }
      }
      // update the cluster center
      clusterCenter(word)(sense).+=(contextsEmbedding)
      clusterCount(word)(sense) += 1
      sense
  }
  
  // Use the cluster centers to predict the sense. Very similar to DP-Means (kulis and jordon)
  def cbow_predict_dpmeans(word: Int, contexts: Seq[Int]): Int = {
      val contextsEmbedding = new DenseTensor1(D, 0)    
      contexts.foreach(context => contextsEmbedding.+=(global_weights(context).value))
      var sense = 0
      
      if (learnMultiVec(word)) {
        var minDist = Double.MaxValue
        val nC = if (ncluster(word) == S) S else ncluster(word) + 1
        var prob = new Array[Double](nC)
        for (s <- 0 until ncluster(word)) {
          val mu = clusterCenter(word)(s) / (clusterCount(word)(s))
          val dist = 1 - TensorUtils.cosineDistance(contextsEmbedding, mu) 
          prob(s) = dist
          if (dist < minDist) {
            minDist = dist
            sense = s
          }
      }
      // create a new cluster only if # of current of senses is less than S(Max-number of senses)
      if (ncluster(word) < S) {
        if (createClusterlambda < minDist) {
          prob(ncluster(word)) = createClusterlambda
          sense = ncluster(word)
          ncluster(word) += 1
        }
      }
    }
    // update the cluster center
    clusterCenter(word)(sense).+=(contextsEmbedding)
    clusterCount(word)(sense) += 1
    sense
  }
  
  // subsampling 
  def subSample(word: Int): Int = {
    val ran = vocab.getSubSampleProb(word) // see the vocabBuilder to understand how this sub-sample prob is got
    val real_ran = rng.nextInt(0xFFFF) / 0xFFFF.toDouble
    return if (ran < real_ran) -1 else word
  }
}



class MSCBOWSkipGramNegSamplingExample(model: MultiSenseWordEmbeddingModel, word: Int, sense : Int, context : Int, label: Int) extends Example {

  // to understand the gradient and objective refer to : http://arxiv.org/pdf/1310.4546.pdf
  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
    
    val wordEmbedding = model.sense_weights(word)(sense).value
    val contextEmbedding = model.global_weights(context).value
    
    
    val score: Double = wordEmbedding.dot(contextEmbedding)
    val exp: Double = math.exp(-score) // TODO : pre-compute expTable similar to word2vec

    var objective: Double = 0.0
    var factor: Double = 0.0
    
    // for POS Label
    if (label == 1) {
      objective = -math.log1p(exp) // log1p -> log(1+x)
      factor = exp / (1 + exp)
    }
    // for NEG Label
    if (label == -1) {
      objective = -score - math.log1p(exp)
      factor = -1 / (1 + exp)
    }
    
    if (value ne null) value.accumulate(objective)
    if (gradient ne null) {
     gradient.accumulate(model.sense_weights(word)(sense), contextEmbedding, factor)
     // don;t update if global_weights is fixed. 
     if (model.updateGlobal == 1) gradient.accumulate(model.global_weights(context), wordEmbedding, factor)
    }

  }
}

// TODO: class MSCBOWSkipGramWsabieExample -> Wsabie-style ranking loss
