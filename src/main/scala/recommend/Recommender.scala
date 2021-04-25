package recommend

import org.rogach.scallop._
import org.json4s.jackson.Serialization
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level
import similarity.Predictor.{normalDevi, pui}

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val data = opt[String](required = true)
  val personal = opt[String](required = true)
  val json = opt[String]()
  verify()
}

case class Rating(user: Int, item: Int, rating: Double)

object Recommender extends App {
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
  println("Loading data from: " + conf.data())
  val dataFile = spark.sparkContext.textFile(conf.data())
  val data = dataFile.map(l => {
      val cols = l.split("\t").map(_.trim)
      Rating(cols(0).toInt, cols(1).toInt, cols(2).toDouble)
  })
  assert(data.count == 100000, "Invalid data")

  println("Loading personal data from: " + conf.personal())
  val personalFile = spark.sparkContext.textFile(conf.personal())
  // TODO: Extract ratings and movie titles
  assert(personalFile.count == 1682, "Invalid personal data")

  // ***** my code starts here ******
  val myUserId = 944
  val parsed = personalFile.map(l => l.split(",").map(_.trim))
  val myRatings = parsed
    .filter(cols=>cols.length==3)
    .map(cols => Rating(myUserId, cols(0).toInt, cols(2).toDouble))
  val itemNameMap = parsed
    .filter(cols=>cols.length!=3)
    .map(cols => (cols(0).toInt, cols(1)))
  val test = parsed
    .filter(cols=>cols.length!=3)
    .map(cols => Rating(myUserId, cols(0).toInt, -1.0))

  // train on all data available, and predict all movies that haven't been rated by me
  val train = data++myRatings

  val avgGlobal = train.map(r => r.rating).mean
  val ru_s = train.groupBy(r => r.user).map{
    case (user, rs) => (user, rs.map(r=>r.rating).sum / rs.size.toDouble)
  }  // (u, ru_)

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


  def get_knnSims() = {
    val temp = cosSims.groupBy(suv=>suv._1._1) // (u,[((u,v),suv),...])
        .map(   ts=>ts._2.toList.sortWith( (t1,t2)=>t1._2>t2._2 ).take(300)  )
    val k300nnSims = temp.flatMap(t=>t)
    val k30nnSims = temp.flatMap(t=>t.take(30))
    (k300nnSims, k30nnSims)
  }
  //val Ks = Seq(300,30)
  val (k300nnSims, k30nnSims) = get_knnSims()


  // ***** equation 2, weighted sum deviation *****
  // rbarhats for all the (u,i) s.t. i is in train and hence rbarhat_i(u) is defined
  val rbarhats_k300nn = get_rbarhats(k300nnSims, train)
  val rbarhats_k30nn = get_rbarhats(k30nnSims, train)
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
  def get_rPred(rbarhats:RDD[((Int,Int),Double)]) = {
    test.map(r=>((r.user, r.item),1)).leftOuterJoin(rbarhats) // ((u,i),(1,Option(rbarhat))
      .map({  case((u,i),t) => (u,(i,{if (t._2.isEmpty) 0 else t._2.get}))  }) // (u, (i,rbarhat_ui/0)
      .leftOuterJoin(ru_s)  // (u,((i,rbarhat_ui/avgGlobal),option(ru_))), a user in test might not be in train
      .map(   {case( u, ((i,rbarhat_ui),ru_) ) =>
        ((u,i),pui({if (ru_.isDefined) ru_.get else avgGlobal},rbarhat_ui))}   )
  }

  // get the top5 items itemIds and ratings for K=300 and K=30
  val top5_k300nn = get_rPred(rbarhats_k300nn)
    .map{case((u,i),rating) => (rating, -i)} // prefer higher rating and smaller itemId
    .top(5)
    .map{case(rating, negI)=>(rating, -negI)}
  val finalListK300 = top5ToFinalList(top5_k300nn)

  val top5_k30nn = get_rPred(rbarhats_k30nn)
    .map{case((u,i),rating) => (rating, -i)} // prefer higher rating and smaller itemId
    .top(5)
    .map{case(rating, negI)=>(rating, -negI)}
  val finalListK30 = top5ToFinalList(top5_k30nn)

  // turn the rdd stores top 5 recommended items to the form of the final output
  def top5ToFinalList(top: Array[(Double, Int)]):List[Any] = {
    var finalList = List[Any]()
    for (i <- top.indices) {
      var entryList = List[Any]()
      entryList = entryList :+ top(i)._2 :+ itemNameMap.lookup(top(i)._2).head:+ top(i)._1 //fixme itemNameMap not right ratings aren't right
      finalList = finalList :+ entryList
    }
    finalList
  }


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

          // IMPORTANT: To break ties and ensure reproducibility of results,
          // please report the top-5 recommendations that have the smallest
          // movie identifier.

          "Q3.2.5" -> Map(
            "Top5WithK=30" ->
              finalListK30,
//              List[Any](
//                List(0, "", 0.0), // Datatypes for answer: Int, String, Double
//                List(0, "", 0.0), // Representing: Movie Id, Movie Title, Predicted Rating
//                List(0, "", 0.0), // respectively
//                List(0, "", 0.0),
//                List(0, "", 0.0)
//              ),

            "Top5WithK=300" ->
              finalListK300
//              List[Any](
//                List(0, "", 0.0), // Datatypes for answer: Int, String, Double
//                List(0, "", 0.0), // Representing: Movie Id, Movie Title, Predicted Rating
//                List(0, "", 0.0), // respectively
//                List(0, "", 0.0),
//                List(0, "", 0.0)
//              )

            // Discuss the differences in rating depending on value of k in the report.
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
