package similarity

import org.rogach.scallop._
import org.json4s.jackson.Serialization
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

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
  // fixme maybe rcs or sim should only be calculated for (u,i) in test?
  val rcs = rhat_ui.map{case((u,i),rhat_ui) => (u, (i, rhat_ui))}
    .join(denoms)
    .map{case(u, ((i, rhat_ui),denom)) => ((u,i), rhat_ui/denom)}
  //println(s"rcs = ${   rcs.map( {case((u,i), rc)=>rc} ).stats()   }")
  // end of preprocessing


  // ***** equation 5, calculate cosine similarities *****
  // get nonzero cosine similarities for all u,v that have shared i
  // (i,u) join (i,v) on i generate (i,(u,v)), where i rated by both u,v
  // then group by (u,v), get ((u,v),[i1,i2,...]) where i1,i2,... are rated by both u,v
  val rcs_byI = rcs.map({case((u,i),rc)=>(i,(u,rc))})  // ()
  val cosSims = rcs_byI.join(rcs_byI) //all triplets ( i, ((u,rc_ui),(v,rc_vi)) ) such that i is rated by both u,v
    .map({case( i, ((u, rc_ui),(v,rc_vi)) ) => ((u,v),rc_ui*rc_vi)})
    .reduceByKey(_+_)
//  println("sims stats:")
//  println(cosSims.values.stats())


  // ***** equation 2, weighted sum deviation *****
  // rbarhats for all the (u,i) s.t. i is in train and hence rbarhat_i(u) is defined
  val rbarhats_cosSims = get_rbarhats(cosSims, train)
  def get_rbarhats(sims: RDD[((Int,Int),Double)], train:RDD[Rating]) = {
    val trGroupbyI = train.map(r=>(r.item, r.user)).groupByKey() // (i, [v1,v2,...])
    val temp = test.map(r=>(r.item, r.user)).join(trGroupbyI) // (i',(u',[v1,v2,...]))
      .flatMap({case(i,(u,vs)) => vs.map( v=>((v,i),u)  )}) // ((v,i'),u'), i is rated by both u',v
      .join(rhat_ui).map({case( (v,i),(u, r_vi) ) => ((u,v),(i,r_vi))}) // ((u',v),(i, r_vi))
      // (u',v): u' test's user, v has rated i' todo can (u', v) be not in sim? yeah..u'might not be in train
      //.map({case((u,v),(i,r_vi))=>((u,v),(i,r_vi, simDum))}) // this for pretending suv=1
      .leftOuterJoin(sims) // ( (u,v),((i, r_vi),suv) )
      temp.map({case((u,v),((i,r_vi), Some(suv))) => ((u,i),(suv*r_vi,math.abs(suv)))
      case((u,v),((i,r_vi), None)) => ((u,i),(0.0,0.0))})  //((u,i),[(suv*rvi, |suv|)])
      .reduceByKey((t1,t2)=>(t1._1+t2._1, t1._2+t2._2)) // ((u,i), (sum(suv*rvi), sum(|suv|)))
      .map({case((u,i),t)=>((u,i),t._1/t._2)}) //((u,i), rbarhat_ui))
  }

  // ***** equation 3, generate predictions *****
  val rPred_cosSim = get_rPred(rbarhats_cosSims)
  def get_rPred(rbarhats:RDD[((Int,Int),Double)]) = {
    test.map(r=>((r.user, r.item),1)).leftOuterJoin(rbarhats) // ((u,i),(1,Option(rbarhat))
      .map({  case((u,i),t) => (u,(i,{if (t._2.isEmpty) 0 else t._2.get}))  }) // (u, (i,rbarhat_ui/0)
      .leftOuterJoin(ru_s)  // (u,((i,rbarhat_ui/avgGlobal),option(ru_))), a user in test might not be in train
      .map(   {case( u, ((i,rbarhat_ui),ru_) ) =>
        ((u,i),pui({if (ru_.isDefined) ru_.get else avgGlobal},rbarhat_ui))}   )
  }

  // ***** calculate mae for method using cosine similarities *****
  val rTrue = test.map(r=>((r.user, r.item), r.rating))
  val cosMae = maeUIR(rTrue, rPred_cosSim)

  //---------------------------------------------------------------------------------
  // ***** calculate jaccard similarities *****

  // similar to assignment, we compare the overlapping between the sets of items "LIKED" by u and v
  val countByU = train.map(r=>(r.user, r.item)).groupByKey().mapValues(is=>is.size)  // [(u, [i1,i2,...])]
  val posRatingsByI = train.map(r=>(r.item, r.user))
  val intersectCounts = posRatingsByI.join(posRatingsByI) // all triplets (i, (u,v)) s.t. i rated by both (u,v).
    .map({case(i, (u,v)) => ((u,v),1)})
    .reduceByKey(_+_)
  val jacSims = intersectCounts.map({case((u,v),uv_count) => (u, (v, uv_count))}).join(countByU)  //(u, ((v, uv_count),u_count))
    .map({case(u,((v, uv_count),u_count))=>(v,(u,uv_count,u_count))}).join(countByU)//(v, ((u,uv_count,u_count),v_count)
    .map({case(v, ((u,uv_count,u_count),v_count)) => (  (u,v), uv_count.toDouble/(u_count+v_count-uv_count).toDouble )})

  val rbarhats_jacSims = get_rbarhats(jacSims, train)     // equation 2

  val rPred_jacSim = get_rPred(rbarhats_jacSims)   // euqation 3

  val jacMae = maeUIR(rTrue, rPred_jacSim)


  // *** helper functions

  // both rTrue and rPred are in the form of ((u, i), r), where (u,i) is the unique key
  // rTure's r is the actual rating, rPred's r is the predicted rating
  def maeUIR(rTrue:RDD[((Int, Int), Double)], rPred:RDD[((Int, Int), Double)]): Double = {
    assert(rTrue.count == rPred.count)
    val joined = rTrue.join(rPred)  // ((u,i), (r_true, r_pred))
    val maeRdd0 = joined.map{
      case((u,i), (r_true, r_pred))=>scala.math.abs(r_true-r_pred)
    }
    assert(rTrue.count == maeRdd0.count)
    return maeRdd0.mean
  }

  def optionalPui(ru:Option[Double], rbarhat_ui:Double, avgGlobal:Double):Double = {
    if (ru.isEmpty) {
      return pui(avgGlobal, rbarhat_ui)
    } else {
      return pui(ru.get, rbarhat_ui)
    }
  }

  // generate a prediction for (u,i) using ru_ and rbarhat_i
  def pui(ru:Double, rbarhat_i:Double):Double = {
    ru + rbarhat_i * scale((ru+rbarhat_i), ru)
  }

  // normalizedDeviation
  def normalDevi(rui:Double, ru:Double): Double = {
    (rui - ru)/scale(rui, ru)
  }

  def scale(x:Double, ru:Double): Double = {
    if (x>ru) {
      5-ru
    } else if (x<ru) {
      ru-1
    } else {
      1
    }
  }

  //---------------------------------------------------------------------------------

  // ***** Q2.3.3 *****
  val numUser = ru_s.count()

  // ***** Q2.3.4 *****
  // number of sim to compute = number of intersections |I(u) intersect I(v)|
  // there are in total numUser^2 (u,v) pairs. We want the statistics of all |I(u) intersect I(v)|
  // intersectCounts are non-zero |I(u) intersect I(v)|. For all other u,v, |I(u) intersect I(v)|=0
  val numZero = numUser*numUser - intersectCounts.count
  val allIntersect = intersectCounts.values.union(spark.sparkContext.parallelize(Range(0,numZero.toInt).map(_=>0)))
  val min = allIntersect.min
  val max = allIntersect.max
  val mean = allIntersect.mean
  val stdDev = allIntersect.stdev

  // ***** Q2.3.5 *****
  val numNonzeroSims = cosSims.count.toInt
  val numBytesNonzero = 8*numNonzeroSims //double is of size 8 bytes

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
          "Q2.3.1" -> Map(
            "CosineBasedMae" -> cosMae, // Datatype of answer: Double
            "CosineMinusBaselineDifference" -> (cosMae-0.7669) // Datatype of answer: Double
          ),

          "Q2.3.2" -> Map(
            "JaccardMae" -> jacMae, // Datatype of answer: Double
            "JaccardMinusCosineDifference" -> (jacMae - cosMae) // Datatype of answer: Double
          ),

          "Q2.3.3" -> Map(
            // Provide the formula that computes the number of similarity computations
            // as a function of U in the report.
            "NumberOfSimilarityComputationsForU1BaseDataset" ->  numUser*numUser // Datatype of answer: Int
          ),

          "Q2.3.4" -> Map(
            "CosineSimilarityStatistics" -> Map(
              "min" -> min,  // Datatype of answer: Double
              "max" -> max, // Datatype of answer: Double
              "average" -> mean, // Datatype of answer: Double
              "stddev" -> stdDev // Datatype of answer: Double
            )
          ),

          "Q2.3.5" -> Map(
            // Provide the formula that computes the amount of memory for storing all S(u,v)
            // as a function of U in the report.
            "TotalBytesToStoreNonZeroSimilarityComputationsForU1BaseDataset" -> numBytesNonzero // Datatype of answer: Int
          ),

          "Q2.3.6" -> Map(
            "DurationInMicrosecForComputingPredictions" -> Map(
              "min" -> 0.0,  // Datatype of answer: Double
              "max" -> 0.0, // Datatype of answer: Double
              "average" -> 0.0, // Datatype of answer: Double
              "stddev" -> 0.0 // Datatype of answer: Double
            )
            // Discuss about the time difference between the similarity method and the methods
            // from milestone 1 in the report.
          ),

          "Q2.3.7" -> Map(
            "DurationInMicrosecForComputingSimilarities" -> Map(
              "min" -> 0.0,  // Datatype of answer: Double
              "max" -> 0.0, // Datatype of answer: Double
              "average" -> 0.0, // Datatype of answer: Double
              "stddev" -> 0.0 // Datatype of answer: Double
            ),
            "AverageTimeInMicrosecPerSuv" -> 0.0, // Datatype of answer: Double
            "RatioBetweenTimeToComputeSimilarityOverTimeToPredict" -> 0.0 // Datatype of answer: Double
          )
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
