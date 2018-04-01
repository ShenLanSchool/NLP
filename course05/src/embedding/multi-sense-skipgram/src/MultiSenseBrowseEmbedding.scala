// package cc.factorie.app.nlp.embeddings
import scala.io.Source
import cc.factorie.la.DenseTensor1
import scala.collection.mutable.PriorityQueue
import java.util.zip.GZIPInputStream
import java.io.FileInputStream
import cc.factorie.util.CmdOptions


class BrowseOptions extends CmdOptions {
   val embedding = new CmdOption("embedding", "", "STRING", "use <string> for filename") 
   val top       = new CmdOption("num-neighbour", 20, "INT", "use <int> to specify the k in knn")
}

object MultiSenseEmbeddingBrowse {

  var threshold = 0
  var vocab         = Array[String]()
  var weights       = Array[Array[DenseTensor1]]()
  var ncluster      = Array[Int]()
  var nclusterCount = Array[Array[Int]]()
  var D = 0
  var V = 0
  var top = 25
  var S = 0
  var maxoutMethod = 0
  def main(args: Array[String]) {
    val opts  = new BrowseOptions()
    opts.parse(args)
    val embeddingFile       = opts.embedding.value
    top                     = opts.top.value
    load(embeddingFile)
    //if (args(3).toInt == 1) 
    //   displayKNN()          // given the context words, hard pick the sense and show the K-NN
    //else 
       play()                // given the word, show the K-NN of all the words
  }

  def load(embeddingsFile: String): Unit = {
    var lineItr = embeddingsFile.endsWith(".gz") match {
      case false => Source.fromFile(embeddingsFile).getLines
      case true => io.Source.fromInputStream(new GZIPInputStream(new FileInputStream(embeddingsFile)), "iso-8859-1").getLines
    }
    // first line is (# words, dimension)
    val details = lineItr.next.stripLineEnd.split(' ').map(_.toInt)
    V = if (threshold > 0 && details(0) > threshold) threshold else details(0)
    D = details(1)
    S = details(2)
    maxoutMethod = details(3)
    println("# words : %d , # size : %d".format(V, D))
    vocab = new Array[String](V)
    weights = Array.ofDim[DenseTensor1](V, S+1)
    ncluster = Array.ofDim[Int](V)
    nclusterCount = Array.ofDim[Int](V, S)
    for (v <- 0 until V) {
      val line = lineItr.next.stripLineEnd.split(' ')
      vocab(v) = line(0).toLowerCase
      ncluster(v) = if (line.size > 1) line(1).toInt else S
      if (line.size > 2) {
        for (i <- 0 until ncluster(v)) 
          nclusterCount(v)(i) = line(i + 2).toInt
      }
      for (s <- 0 until ncluster(v) + 1) {
        val line = lineItr.next.stripLineEnd.split(' ')
        weights(v)(s) = new DenseTensor1(D, 0) // allocate the memory
        for (d <- 0 until D) weights(v)(s)(d) = line(d).toDouble
        weights(v)(s) /= weights(v)(s).twoNorm
        if (s > 0 && maxoutMethod == 0) {
            val lineSkip = lineItr.next.stripLineEnd.split(' ')
        }
      }
    }
    println("loaded vocab and their embeddings")
  }
  def play(): Unit = {

    while (true) {
      print("Enter word (EXIT to break) : ")
      var word = readLine.stripLineEnd
      val id  = getID(word)
      println("Id in the vocab for the word " + word + " " + id)
      if (id == -1) {
        println("words not in vocab")
      } else 
      {
        
        for (is <- 0 until ncluster(id)+ 1) 
        {
          val embedding_in = weights(id)(is) // give out nearest neighbours for all the words
          var pq = new PriorityQueue[(String, Double)]()(dis)
          for (i <- 0 until vocab.size) 
          {  
             if (is > 0) {
             for (s <- 1 until ncluster(i) + 1)  
             {
                  val embedding_out = weights(i)(s) // take only senses 
                 val score = TensorUtils.cosineDistance(embedding_in, embedding_out)
                 if (pq.size < top) pq.enqueue(vocab(i) -> score)
                 else if (score > pq.head._2) 
                 { // if the score is greater the min, then add to the heap
                  pq.dequeue
                  val vocab_str = vocab(i) //+ (if (ncluster(i) > 1) "_" + s.toString else "");
                  pq.enqueue(vocab_str -> score)
                 }
             }
             
             }
             if (is ==0) {
                val embedding_out = weights(i)(0) // take only senses 
                 val score = TensorUtils.cosineDistance(embedding_in, embedding_out)
                 if (i < top) pq.enqueue(vocab(i) -> score)
                 else if (score > pq.head._2) 
                 { // if the score is greater the min, then add to the heap
                  pq.dequeue
                  pq.enqueue(vocab(i) -> score)
                 }
             }
          }
        var arr = new Array[(String, Double)](pq.size)
        var i = 0
        while (!pq.isEmpty) 
        { // min heap
          arr(i) = (pq.head._1, pq.head._2)
          i += 1
          pq.dequeue
        }
        print("\t\t\t\t\t\tWord\t\tCosine Distance")
          if (is==0) {
        print("(Global Embedding)") 
          }
          print("\n") 
       arr.reverse.foreach(x => println("%50s\t\t%f".format(x._1, x._2)))

      }
        }
    }
  }
  def playMS(): Unit = {
     while (true) {
       print("Enter a word or sentence to exit : ")
       val words = readLine.stripLineEnd.split(' ').filter(word => getID(word) != -1)
       val int_words = words.map(word => getID(word)).filter(id => id!= -1)
       val embedding_in = new DenseTensor1(D, 0)
       for (i <- 0 until words.size) {
             val w = int_words(i)
             val word = words(i)
             val contexts = int_words.filter(id => id != w)
             val sense = getSense(w, contexts)  
             embedding_in.+=( weights(w)(sense) ) // sum up the local contexts find the closest word
             val arr = knn( weights(w)(sense) )
             print("word : " + sense + " " + word + " -")
             arr.foreach(x => print(" " + x._1 ))
             println()
        
       }
       val arr = knn(embedding_in)
       println("\t\t\t\t\t\tWord\t\tCosine Distance")
       arr.foreach(x => println("%50s\t\t%f".format(x._1, x._2)))
       println()
     }
  }
  
  def displayKNN(): Unit = {
    while (true) {
      print("Enter the word : ")
      val in_word = readLine.stripLineEnd
      val w       = getID(in_word)
      print("Enter the contexts : ")
      val words = readLine.stripLineEnd.split(' ').filter(word => getID(word) != -1)
      val int_words = words.map(word => getID(word)).filter(id => id != -1)
      val contextEmbedding = new DenseTensor1(D)
      (0 until int_words.size).foreach(i => contextEmbedding.+=( weights(int_words(i))(0))    )
      contextEmbedding./=( contextEmbedding.twoNorm )
      val prob = new Array[Double](S+1)
      var z = 0.0
      for (s <- 1 until ncluster(w)+1) {
           val v = contextEmbedding.dot( weights(w)(s) )
           prob(s) = math.exp(v) * math.exp(v)
           //prob(s) = 1/(1+math.exp( -contextEmbedding.dot(weights(w)(s)) ))
           z += prob(s)
      }
      val emb = new DenseTensor1(D)
      for (s <- 1 until ncluster(w)+1) {
           println("Prob- " + s + " " + prob(s)/z)
           val x  = weights(w)(s) * prob(s)/z // multiply with the probablity
           emb.+=( x )
      }
      val nn = knn(emb, 1, S+1, 40)
       println("\t\t\t\t\t\tWord\t\tCosine Distance")
       nn.foreach(x => println("%50s\t\t%f".format(x._1, x._2)))
       println()
    }
    
  }
  // private helper functions
  private def dis() = new Ordering[(String, Double)] {
    def compare(a: (String, Double), b: (String, Double)) = -a._2.compare(b._2)
  }
  private def getID(word: String): Int = {
    for (i <- 0 until vocab.length) if (vocab(i).equalsIgnoreCase(word))
      return i
    return -1
  }
  private def getSense(word : Int, contexts : Seq[Int]): Int =  {
        val contextEmbedding = new DenseTensor1(D, 0)
        (0 until contexts.size).foreach(i => contextEmbedding.+=(weights(contexts(i))(0)) ) // global context
        var correct_sense = 0
        var max_score = Double.MinValue
        for (s <- 1 until ncluster(word)+1) {
             val score = contextEmbedding.dot( weights(word)(s) )// find the local context
             if (score > max_score) {
               correct_sense = s
               max_score = score
             }
             print(s + " " + score + " " + logit(score) + ":")
        }
        correct_sense
  }
  private def logit(x : Double):Double = 1/(1 + math.exp(-x))
  private def knn(in : DenseTensor1, st : Int = 1, en : Int = S+1, t : Int =10): Array[(String, Double)] = {
     var pq = new PriorityQueue[(String, Double)]()(dis)
     for (i <- 0 until vocab.size) 
     {  
       for (s <- 1 until ncluster(i)+1)  
       {
         val out = weights(i)(s) // take only senses 
         val score = TensorUtils.cosineDistance(in, out)
         if (i < top) pq.enqueue(vocab(i) -> score)
         else if (score > pq.head._2) 
         { // if the score is greater the min, then add to the heap
                  pq.dequeue
                  pq.enqueue(vocab(i) -> score)
         }
        }
      }
       var arr = new Array[(String, Double)](pq.size)
       var i = 0
       while (!pq.isEmpty) 
       { // min heap
          arr(i) = (pq.head._1, pq.head._2)
          i += 1
          pq.dequeue
       }
       arr = arr.reverse
       
      arr
   }
}
