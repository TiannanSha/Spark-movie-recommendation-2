package knn

import org.rogach.scallop._
import org.json4s.jackson.Serialization
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level
import similarity.Predictor.{MAE, normalDevi, pui, rbarhats_jacSims, test, train}

import scala.util.control.Breaks.break //todo import cosSims
//import similarity.Rating

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val json = opt[String]()
  verify()
}

case class Rating(user: Int, item: Int, rating: Double)

object Predictor extends App {
  // Remove these lines if encountering/debugging Spark
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark = SparkSession.builder()
    .master("local[1]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  println("")
  println("******************************************************")

  var conf = new Conf(args)
  println("Loading training data from: " + conf.train())
  val trainFile = spark.sparkContext.textFile(conf.train())
  val train = trainFile.map(l => {
      val cols = l.split("\t").map(_.trim)
      Rating(cols(0).toInt, cols(1).toInt, cols(2).toDouble)
  })
  assert(train.count == 80000, "Invalid training data")

  println("Loading test data from: " + conf.test())
  val testFile = spark.sparkContext.textFile(conf.test())
  val test = testFile.map(l => {
      val cols = l.split("\t").map(_.trim)
      Rating(cols(0).toInt, cols(1).toInt, cols(2).toDouble)
  })
  assert(test.count == 20000, "Invalid test data")

  // ***************** my code starts here ***********************
  val avgGlobal = train.map(r => r.rating).mean
  val ru_s = train.groupBy(r => r.user).map{
    case (user, rs) => (user, rs.map(r=>r.rating).sum / rs.size.toDouble)
  }  // (u, ru_)

  // ***** equation 4, preprocessing *****
  // numerator for equation 4
  val rhat_ui = train.map(r=>(r.user, (r.item, r.rating)))   // entry: (u, (i, rui))
    .join(ru_s)  // entry (u, ((i, rui), ru_))
    .map{case(  u, ((i, rui),ru_)  ) => ( (u,i), normalDevi(rui,ru_) )} // ((u,i), rhat_ui)
  // denominator for equation 4
  val denoms = rhat_ui.map{ case ((u,i), rhat) => (u, rhat*rhat)}
    .reduceByKey((a,b)=>a+b)  // (u, sum(rhat_u_^2))
    .mapValues(math.sqrt)
  // equation 4, rc_ui for all (u,i) in train, rc is the preprocessing results
  val rcs = rhat_ui.map{case((u,i),rhat_ui) => (u, (i, rhat_ui))}
    .join(denoms)
    .map{case(u, ((i, rhat_ui),denom)) => ((u,i), rhat_ui/denom)}


  // ***** equation 5, calculate cosine similarities *****
  // get nonzero cosine similarities for all u,v that have shared i
  val rcs_byI = rcs.map({case((u,i),rc)=>(i,(u,rc))})  // ()
  val cosSims = rcs_byI.join(rcs_byI) //all triplets ( i, ((u,rc_ui),(v,rc_vi)) ) such that i is rated by both u,v
    .map({case( i, ((u, rc_ui),(v,rc_vi)) ) => ((u,v),rc_ui*rc_vi)})
    .reduceByKey(_+_)

  // get similarities for different K values, from large to small
  def get_knnSims(Ks:Seq[Int]) = {
    val temp = cosSims.groupBy(suv=>suv._1._1) // (u,[((u,v),suv),...])
      .map(   ts=>ts._2.toList.sortWith( (t1,t2)=>t1._2>t2._2 ).take(Ks(0))  )
    var simsForDiffKs = Seq[RDD[((Int,Int),Double)]]()

    for (k <- Ks) {//fixme delete colesce
      simsForDiffKs = simsForDiffKs :+ temp.flatMap(t=>t.take(k+1)) // including su,u which is never used
    }
    simsForDiffKs
  }

  val Ks = Seq(943,800,400,300,200,100,50,30,10)
  val simsForDiffKs = get_knnSims(Ks)

  // ***** equation 2 and 3, weighted sum deviation and compute predictions *****
  // ***** and then calculate MAEs for different Ks                         *****
  val rTrue = test.map(r=>((r.user, r.item), r.rating))
  var MAE_ForDiffKs = Seq[Double]()
  val trGroupbyI = train.map(r=>(r.item, r.user)).groupByKey() // (i, [v1,v2,...])
  val rhat_reorganised = test.map(r=>(r.item, r.user)).join(trGroupbyI) // (i',(u',[v1,v2,...]))
    .flatMap({case(i,(u,vs)) => vs.map( v=>((v,i),u)  )}) // ((v,i'),u'), i is rated by both u',v
    .join(rhat_ui).map({case( (v,i),(u, r_vi) ) => ((u,v),(i,r_vi))}) // ((u',v),(i, r_vi)) fixme reuse this
  for (sims <- simsForDiffKs) {
    MAE_ForDiffKs = MAE_ForDiffKs :+ MAE(rTrue, get_rPred( get_rbarhats(sims, train, rhat_reorganised)))
  }

  // Q3.2.1 questions about lowest K that has better MAE than the baseline
  val MAE_Baseline = 0.7669
  var minK = 8
  var MAE_ForMinK = 0.7669
  var i = 8
  while (i>=0) {
    if (MAE_ForDiffKs(i) < MAE_Baseline) {
      minK = Ks(i)
      MAE_ForMinK = MAE_ForDiffKs(i)
      i = -1 // found the k, break the loop
    }
    i -= 1
  }

  def get_rbarhats(sims: RDD[((Int,Int),Double)], train:RDD[Rating],
  rhat_reorganized:RDD[((Int,Int),(Int, Double))]) = {
//    val trGroupbyI = train.map(r=>(r.item, r.user)).groupByKey() // (i, [v1,v2,...])
//    test.map(r=>(r.item, r.user)).join(trGroupbyI) // (i',(u',[v1,v2,...]))
//      .flatMap({case(i,(u,vs)) => vs.map( v=>((v,i),u)  )}) // ((v,i'),u'), i is rated by both u',v
//      .join(rhat_ui).map({case( (v,i),(u, r_vi) ) => ((u,v),(i,r_vi))}) // ((u',v),(i, r_vi)) fixme reuse this
      rhat_reorganised
      .leftOuterJoin(sims) // ( (u,v),((i, r_vi),suv) )
      .map({case((u,v),((i,r_vi), Some(suv))) => ((u,i),(suv*r_vi,math.abs(suv)))
            case((u,v),((i,r_vi), None)) => ((u,i),(0.0,0.0))})  //((u,i),[(suv*rvi, |suv|)])
      .reduceByKey((t1,t2)=>(t1._1+t2._1, t1._2+t2._2)) // ((u,i), (sum(suv*rvi), sum(|suv|)))
      .map({case((u,i),t)=>((u,i),if (t._2==0.0) 0.0 else t._1/t._2)}) //((u,i), rbarhat_ui))
  }

  // ***** equation 3, generate predictions *****
  def get_rPred(rbarhats:RDD[((Int,Int),Double)]) = {
    test.map(r=>((r.user, r.item),1)).leftOuterJoin(rbarhats) // ((u,i),(1,Option(rbarhat))
      .map({  case((u,i),t) => (u,(i,{if (t._2.isEmpty) 0.0 else t._2.get}))  }) // (u, (i,rbarhat_ui/0)
      .leftOuterJoin(ru_s)  // (u,((i,rbarhat_ui/avgGlobal),option(ru_))), a user in test might not be in train
      .map(   {case( u, ((i,rbarhat_ui),ru_) ) =>
        ((u,i),pui({if (ru_.isDefined) ru_.get else avgGlobal},rbarhat_ui))}   )
  }

  // ***** Q3.2.2 *****
  val numUser = ru_s.count.toInt
  val minNumBytesKs = Ks.map(k => 8*k*numUser)

  // ***** Q3.2.3 *****
  val RAMSizeBytes:Long = 8000000000L
  // each user needs minK number of sims, each sim has size 3*8 bytes (2 64bit-Integer and 1 Double)
  val userSizeBytes:Long = minK * 24
  val numUserInRAM = RAMSizeBytes / userSizeBytes

  // Save answers as JSON
  def printToFile(content: String,
                  location: String = "./answers.json") =
    Some(new java.io.PrintWriter(location)).foreach{
      f => try{
        f.write(content)
      } finally{ f.close }
  }
  conf.json.toOption match {
    case None => ;
    case Some(jsonFile) => {
      var json = "";
      {
        // Limiting the scope of implicit formats with {}
        implicit val formats = org.json4s.DefaultFormats
        val answers: Map[String, Any] = Map(
          "Q3.2.1" -> Map(
            // Discuss the impact of varying k on prediction accuracy on
            // the report.
            "MaeForK=10" -> MAE_ForDiffKs(8), // Datatype of answer: Double
            "MaeForK=30" -> MAE_ForDiffKs(7), // Datatype of answer: Double
            "MaeForK=50" -> MAE_ForDiffKs(6), // Datatype of answer: Double
            "MaeForK=100" -> MAE_ForDiffKs(5), // Datatype of answer: Double
            "MaeForK=200" -> MAE_ForDiffKs(4), // Datatype of answer: Double
            "MaeForK=300" -> MAE_ForDiffKs(3), // Datatype of answer: Double
            "MaeForK=400" -> MAE_ForDiffKs(2), // Datatype of answer: Double
            "MaeForK=800" -> MAE_ForDiffKs(1), // Datatype of answer: Double
            "MaeForK=943" -> MAE_ForDiffKs(0), // Datatype of answer: Double
            "LowestKWithBetterMaeThanBaseline" -> minK, // Datatype of answer: Int
            "LowestKMaeMinusBaselineMae" -> (MAE_ForMinK - MAE_Baseline) // Datatype of answer: Double
          ),

          "Q3.2.2" ->  Map(
            // Provide the formula the computes the minimum number of bytes required,
            // as a function of the size U in the report.
            "MinNumberOfBytesForK=10" -> minNumBytesKs(8), // Datatype of answer: Int
            "MinNumberOfBytesForK=30" -> minNumBytesKs(7), // Datatype of answer: Int
            "MinNumberOfBytesForK=50" -> minNumBytesKs(6), // Datatype of answer: Int
            "MinNumberOfBytesForK=100" -> minNumBytesKs(5), // Datatype of answer: Int
            "MinNumberOfBytesForK=200" -> minNumBytesKs(4), // Datatype of answer: Int
            "MinNumberOfBytesForK=300" -> minNumBytesKs(3), // Datatype of answer: Int
            "MinNumberOfBytesForK=400" -> minNumBytesKs(2), // Datatype of answer: Int
            "MinNumberOfBytesForK=800" -> minNumBytesKs(1), // Datatype of answer: Int
            "MinNumberOfBytesForK=943" -> minNumBytesKs(0) // Datatype of answer: Int
          ),

          "Q3.2.3" -> Map(
            "SizeOfRamInBytes" -> RAMSizeBytes, // Datatype of answer: Long
            "MaximumNumberOfUsersThatCanFitInRam" -> numUserInRAM // Datatype of answer: Long
          )

          // Answer the Question 3.2.4 exclusively on the report.
         )
        json = Serialization.writePretty(answers)
      }

      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json, jsonFile)
    }
  }

  println("")
  spark.close()
}
