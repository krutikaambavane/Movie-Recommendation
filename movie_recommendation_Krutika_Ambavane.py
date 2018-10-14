import os
import sys
import math
from pyspark import SparkConf, SparkContext
from pyspark import SQLContext
from math import sqrt
from operator import add

DATA_DIR = "/Users/aniketalshi/Downloads/movie_recomendation/ml-latest-small";
conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities")
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

def extract_ratings(ratingsFile):
    # extract the header out
    header = ratingsFile.first()
    dataHeader = sc.parallelize([header])
    ratingsData = ratingsFile.subtract(dataHeader)

    # extract user_id, movie_id, rating
    ratings = ratingsData.map(lambda l : l.split(",")).map(lambda l : (int(l[0]), int(l[1]), float(l[2])))
    return ratings

def extract_movies(movieFile):
    movieNames = {}
    with open(movieFile) as f:
        next(f) # skip the header
        for line in f:
            # movieId, moviename, genre|genre
            if '"' not in line : 
                fields = line.split(',')
                movieNames[int(fields[0])] = fields[1].decode('ascii', 'ignore').strip('"')
            else:
                ''' 
                We cannot simply split on , because it is part of movie name 
                movieID, "movie, name", genre|genre
                '''
                pos = line.find(',') 
                movieID = int(line[0:pos])
                movieName = line[pos+1 : line.rfind(',')].decode('ascii', 'ignore').strip('"')
                movieNames[movieID] = movieName
    return movieNames

def getRatingsWithSize(ratings, numRaters):
    ratings_df = ratings.toDF(["col1", "col2", "col3"])
    raters_df = numRaters.toDF(["col1", "col2"])

    rsz = ratings_df.alias("ratings_df").\
                    join(raters_df.alias("raters_df"), ratings_df.col2==raters_df.col1).\
                    select(ratings_df.col1, ratings_df.col2, ratings_df.col3, raters_df.col2).rdd.map(tuple)
    return rsz


def filterDuplicates( (userID, ratings) ):
    (movie1, rating1, numRatings1) = ratings[0]
    (movie2, rating2, numRatings2) = ratings[1]
    return movie1 < movie2

def makePairs( ratings ):
    (movie1, rating1, numRatings1) = ratings[0], ratings[1], ratings[2]
    (movie2, rating2, numRatings2) = ratings[3], ratings[4], ratings[5]
    return ((movie1, movie2), ((rating1, numRatings1), (rating2, numRatings2)))

def convertToVector( (key, ratings) ) :
    rating1, numRatings1 = ratings[0][0], ratings[0][1]
    rating2, numRatings2 = ratings[1][0], ratings[1][1]

    return (key, (1, rating1 * rating2, rating1, rating2, math.pow(rating1, 2), math.pow(rating2, 2), numRatings1, numRatings2))

def jaccardSimilarity( (moviePair, ratingsVector) ):
    usersInCommon = ratingsVector[0]
    totalUsers1 = ratingsVector[6]
    totalUsers2 = ratingsVector[7]

    return (moviePair, usersInCommon/ float(totalUsers1 + totalUsers2 - usersInCommon))

def correlation( (moviePair, ratingsVector) ):
    size = ratingsVector[0]
    dotProduct = ratingsVector[1]
    ratingSum = ratingsVector[2]
    rating2Sum = ratingsVector[3]

    ratingNormSq = ratingsVector[4]
    rating2NormSq = ratingsVector[5]
    
    numerator = size * dotProduct - ratingSum * rating2Sum
    denominator = math.sqrt(size * ratingNormSq - ratingSum * ratingSum) * float(math.sqrt(size * rating2NormSq - rating2Sum * rating2Sum))
    
    score = 0.0 
    if(denominator):
        score = numerator / float(denominator)

    return (moviePair, score)


def findSimilarMovies(similarities, movieID, scoreThreshold):
    results = similarities.filter(lambda (x, y) : (x[0] == movieID or x[1] == movieID) and y >= scoreThreshold)
    return results.map(lambda (x, y) : (y, x)).sortByKey(ascending=False).take(20)


def printSimilarMovies(movieID, similarities, movies, scoreThreshold):
    similarMovies = findSimilarMovies(similarities, movieID, scoreThreshold)
        
    print("\n\n {} movies similar to {} : ".format(len(similarMovies), movies[movieID]))
    for score, movie in similarMovies:
        (movie1, movie2) = movie
        if movie1 == movieID :
            print movies[movie2], 
        else:
            print movies[movie1], 
        print(" , Similarity Score : {}\n".format(score))

def main() :
    # load ratings data file
    ratingsFile = sc.textFile(DATA_DIR + "/ratings.csv")
    
    # extract userid, movieid, rating from file
    ratings = extract_ratings(ratingsFile)

    # movie file
    movieFile = DATA_DIR + "/movies.csv"
    movies = extract_movies(movieFile)

    # get num raters per movie keyed on movie id
    numRaters = ratings.map(lambda l : (l[1], 1)).reduceByKey(add)
    
    # join two rdds so we have [user_id, movie_id, ratings, num_ratings_for_movie]
    ratingsWithSize = getRatingsWithSize(ratings, numRaters)

    # map of key (userid) => (movie id, rating, numRatings)
    ratingsMap = ratingsWithSize.map(lambda l : (l[0], (l[1], l[2], l[3])))
    
    # Self-join to find every combination.
    joinedRatings = ratingsMap.join(ratingsMap)
    
    # Filter out duplicate pairs and map to (movie1, rating1, num_raters1, movie2, rating2, num_raters2)
    uniqueJoinedRatings = joinedRatings.filter(filterDuplicates).\
                          map(lambda x : (x[1][0][0], x[1][0][1], x[1][0][2], x[1][1][0], x[1][1][1], x[1][1][2]))
    
    # make pair (movie1, movie2) => 
    moviePairs = uniqueJoinedRatings.map(makePairs)

    # convert to it vector of seven key => (1, rating1 * rating2, rating1, rating2, rating1 ^ 2, rating2 ^ 2, num_raters1, rum_raters2)
    movieVector = moviePairs.map(convertToVector)
    
    movieGroupedBy = movieVector.reduceByKey(lambda x, y: (x[0] + y[0],               # size
                                                           x[1] + y[1],               # sum(rating1 * rating2)
                                                           x[2] + y[2],               # sum(rating1)
                                                           x[3] + y[3],               # sum(rating2)
                                                           x[4] + y[4],               # sum(rating1^2)
                                                           x[5] + y[5],               # sum(rating2^2)
                                                           max(x[6], y[6]),           # sum(num_raters1)
                                                           max(x[7], y[7]))).cache()  # sum(num_raters2)
    
    jaccardSimilarities = movieGroupedBy.map(jaccardSimilarity).cache()
    
    # some sample movieIds to test
    movieID = 260
    print("\nUsing jaccard similarity\n")
    printSimilarMovies(movieID, jaccardSimilarities, movies, 0.20)
        
    
if __name__ == "__main__":
    main()
