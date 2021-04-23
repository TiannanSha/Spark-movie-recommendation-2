package knn

import org.rogach.scallop._
import org.json4s.jackson.Serialization
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level
import similarity.Predictor.{maeUIR, normalDevi, pui, rbarhats_jacSims, test, train} //todo import cosSims
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
  // fixme maybe rcs or sim should only be calculated for (u,i) in test?
  val rcs = rhat_ui.map{case((u,i),rhat_ui) => (u, (i, rhat_ui))}
    .join(denoms)
    .map{case(u, ((i, rhat_ui),denom)) => ((u,i), rhat_ui/denom)}
//  println(s"rcs = ${   rcs.map( {case((u,i), rc)=>rc} ).stats()   }")
  // end of preprocessing
  //fixme rc might be bad


  // ***** equation 5, calculate cosine similarities *****
  // get nonzero cosine similarities for all u,v that have shared i
  // (i,u) join (i,v) on i generate (i,(u,v)), where i rated by both u,v
  // then group by (u,v), get ((u,v),[i1,i2,...]) where i1,i2,... are rated by both u,v
  val rcs_byI = rcs.map({case((u,i),rc)=>(i,(u,rc))})  // ()
  val cosSims = rcs_byI.join(rcs_byI) //all triplets ( i, ((u,rc_ui),(v,rc_vi)) ) such that i is rated by both u,v
    .map({case( i, ((u, rc_ui),(v,rc_vi)) ) => ((u,v),rc_ui*rc_vi)})
    .reduceByKey(_+_)

  def get_knnSims(K:Int) = {
    cosSims.groupBy(suv=>suv._1._1) // (u,[((u,v),suv),...])
      .flatMap(   ts=>ts._2.toList.sortWith( (t1,t2)=>t1._2>t2._2 ).take(K+1)  )
    //also take suu, which is always 1 and won't be used
//    cosSims.groupBy(suv=>suv._1._1) // (u,[((u,v),suv),...])
//      .flatMap(ts=>ts._2.toSeq.sortWith((s1,s2)=>s1._2>s1._2).take(k+1))

  }
  val K = 300
  val knnSims = get_knnSims(K)

  // ***** equation 2, weighted sum deviation *****
  // rbarhats for all the (u,i) s.t. i is in train and hence rbarhat_i(u) is defined
  val rbarhats_knnSims = get_rbarhats(knnSims, train)

  def get_rbarhats(sims: RDD[((Int,Int),Double)], train:RDD[Rating]) = {
    val trGroupbyI = train.map(r=>(r.item, r.user)).groupByKey() // (i, [v1,v2,...])
    test.map(r=>(r.item, r.user)).join(trGroupbyI) // (i',(u',[v1,v2,...]))
      .flatMap({case(i,(u,vs)) => vs.map( v=>((v,i),u)  )}) // ((v,i'),u'), i is rated by both u',v
      .join(rhat_ui).map({case( (v,i),(u, r_vi) ) => ((u,v),(i,r_vi))}) // ((u',v),(i, r_vi))
      .leftOuterJoin(sims) // ( (u,v),((i, r_vi),suv) )

      .map({case((u,v),((i,r_vi), Some(suv))) => ((u,i),(suv*r_vi,math.abs(suv)))
            case((u,v),((i,r_vi), None)) => ((u,i),(0.0,0.0))})  //((u,i),[(suv*rvi, |suv|)])
      .reduceByKey((t1,t2)=>(t1._1+t2._1, t1._2+t2._2)) // ((u,i), (sum(suv*rvi), sum(|suv|)))
      .map({case((u,i),t)=>((u,i),if (t._2==0.0) 0.0 else t._1/t._2)}) //((u,i), rbarhat_ui))

//    // filter first before other operations epspecially join
//      .filter(t=>t._2._2.isDefined)
//      .map({case((u,v),((i,r_vi), Some(suv))) => ((u,i),(suv*r_vi,math.abs(suv)))})  //((u,i),[(suv*rvi, |suv|)])
//      .reduceByKey((t1,t2)=>(t1._1+t2._1, t1._2+t2._2)) // ((u,i), (sum(suv*rvi), sum(|suv|)))
//      .map({case((u,i),t)=>((u,i),if (t._2==0.0) 0.0 else t._1/t._2)}) //((u,i), rbarhat_ui))
  }

  // ***** equation 3, generate predictions *****
  val rPred_knnSim = get_rPred(rbarhats_knnSims)
  def get_rPred(rbarhats:RDD[((Int,Int),Double)]) = {
    test.map(r=>((r.user, r.item),1)).leftOuterJoin(rbarhats) // ((u,i),(1,Option(rbarhat))
      .map({  case((u,i),t) => (u,(i,{if (t._2.isEmpty) 0.0 else t._2.get}))  }) // (u, (i,rbarhat_ui/0)
      .leftOuterJoin(ru_s)  // (u,((i,rbarhat_ui/avgGlobal),option(ru_))), a user in test might not be in train
      .map(   {case( u, ((i,rbarhat_ui),ru_) ) =>
        ((u,i),pui({if (ru_.isDefined) ru_.get else avgGlobal},rbarhat_ui))}   )
  }

  // ***** calculate mae for method using cosine similarities *****
  val rTrue = test.map(r=>((r.user, r.item), r.rating))
  val knnMae = maeUIR(rTrue, rPred_knnSim)
  println(s"knnMae K=$K: $knnMae ")

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
            "MaeForK=10" -> 0.8407036862423928, // Datatype of answer: Double
            "MaeForK=30" -> 0.7914221792247477, // Datatype of answer: Double
            "MaeForK=50" -> 0.7749407796360606, // Datatype of answer: Double
            "MaeForK=100" -> 0.7561353222065882, // Datatype of answer: Double
            "MaeForK=200" -> 0.7484528977469215, // Datatype of answer: Double
            "MaeForK=300" -> 0.7469140388149915, // Datatype of answer: Double
            "MaeForK=400" -> 0.7471389103638729, // Datatype of answer: Double
            "MaeForK=800" -> 0.7475383223779437 , // Datatype of answer: Double
            "MaeForK=943" -> 0.7477653398324886, // Datatype of answer: Double
            "LowestKWithBetterMaeThanBaseline" -> 100, // Datatype of answer: Int
            "LowestKMaeMinusBaselineMae" -> (0.7561353222065882-0.7669) // Datatype of answer: Double
          ),

          "Q3.2.2" ->  Map(
            // Provide the formula the computes the minimum number of bytes required,
            // as a function of the size U in the report.
            "MinNumberOfBytesForK=10" -> 0, // Datatype of answer: Int
            "MinNumberOfBytesForK=30" -> 0, // Datatype of answer: Int
            "MinNumberOfBytesForK=50" -> 0, // Datatype of answer: Int
            "MinNumberOfBytesForK=100" -> 0, // Datatype of answer: Int
            "MinNumberOfBytesForK=200" -> 0, // Datatype of answer: Int
            "MinNumberOfBytesForK=300" -> 0, // Datatype of answer: Int
            "MinNumberOfBytesForK=400" -> 0, // Datatype of answer: Int
            "MinNumberOfBytesForK=800" -> 0, // Datatype of answer: Int
            "MinNumberOfBytesForK=943" -> 0 // Datatype of answer: Int
          ),

          "Q3.2.3" -> Map(
            "SizeOfRamInBytes" -> 0, // Datatype of answer: Long
            "MaximumNumberOfUsersThatCanFitInRam" -> 0 // Datatype of answer: Long
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
