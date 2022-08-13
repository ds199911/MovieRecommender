# DSGA1004 - BIG DATA
## Final project

Contributors: Lucas Tao (ct1942), Billy Sun (ds5749), Cecilia Wu (czw206)

# Final Checkpoint Brief 

- [X] 2022/04/22: popularity baseline model and evaluation on small subset.
- [X] **2022/04/29**: checkpoint submission with baseline results on both small and large datasets.  Preliminary results for matrix factorization on the small dataset.

## Partition 

We split both small and large `ratings.csv` datasets into 70% training, 15% testing, and 15% validation according to timestamp. The first 70% of the orginal data of all users are put into training. The remaining 30% of original data are split into disjoint-user sets of testing and validation. A sample of our small datsets after split can be found in the folder 'data'.

## Baseline 

### Baseline Design 

We implemented the popularity model demonstrated in class by first computing the average utility for each item movieId and then rank item by descending order of `P[i]`. This helped us produce the same popular 100 movies for all users. A sample of the top 20 movie recommendations is given below for both small and large sets. The full lists of 100 movies are saved as `{file size}_{file type}_result.csv` for the respective file in this repository. 

Popular 20 with samll: 

|index|movieId|title|genres|
|---|---|---|---|
|0|14|Nixon \(1995)|Drama|
|1|163|Desperado \(1995)|Action&#124;Romance|Western|
|2|219|Cure, The \(1995)|Drama|
|3|234|Exit to Eden \(1994)|Comedy|
|4|10|GoldenEye \(1995)|Action&#124;Adventure|Thriller|
|5|5|Father of the Bride Part II \(1995)|Comedy|
|6|314|Secret of Roan Inish, The \(1994)|Children&#124;Drama|Fantasy|Mystery|
|7|52|Mighty Aphrodite \(1995)|Comedy&#124;Drama|Romance|
|8|347|Bitter Moon \(1992)|Drama&#124;Film-Noir|Romance|
|9|337|What's Eating Gilbert Grape \(1993)|Drama|
|10|345|Adventures of Priscilla, Queen of the Desert, The \(1994)|Comedy&#124;Drama|
|11|731|Heaven's Prisoners \(1996)|Crime&#124;Thriller|
|12|54|Big Green, The \(1995)|Children&#124;Comedy|
|13|19|Ace Ventura: When Nature Calls \(1995)|Comedy|
|14|338|Virtuosity \(1995)|Action&#124;Sci-Fi|Thriller|
|15|83|Once Upon a Time\... When We Were Colored (1995)|Drama&#124;Romance|
|16|233|Exotica \(1994)|Drama|
|17|207|Walk in the Clouds, A \(1995)|Drama&#124;Romance|
|18|308|Three Colors: White \(Trzy kolory: Bialy) (1994)|Comedy&#124;Drama|
|19|237|Forget Paris \(1995)|Comedy&#124;Romance|

Popular 20 with large: 
|index|movieId|title|genres|
|---|---|---|---|
|0|85|Angels and Insects \(1995)|Drama&#124;Romance|
|1|81|Things to Do in Denver When You're Dead \(1995)|Crime&#124;Drama|Romance|
|2|94|Beautiful Girls \(1996)|Comedy&#124;Drama|Romance|
|3|125|Flirting With Disaster \(1996)|Comedy|
|4|282|Nell \(1994)|Drama|
|5|607|Century \(1993)|Drama|
|6|115|Happiness Is in the Field \(Bonheur est dans le pré, Le) (1995)|Comedy|
|7|316|Stargate \(1994)|Action&#124;Adventure|Sci-Fi|
|8|637|Sgt\. Bilko (1996)|Comedy|
|9|110|Braveheart \(1995)|Action&#124;Drama|War|
|10|56|Kids of the Round Table \(1995)|Adventure&#124;Children|Comedy|Fantasy|
|11|164|Devil in a Blue Dress \(1995)|Crime&#124;Film-Noir|Mystery|Thriller|
|12|31|Dangerous Minds \(1995)|Drama|
|13|169|Free Willy 2: The Adventure Home \(1995)|Adventure&#124;Children|Drama|
|14|146|Amazing Panda Adventure, The \(1995)|Adventure&#124;Children|
|15|47|Seven \(a.k.a. Se7en) (1995)|Mystery&#124;Thriller|
|16|430|Calendar Girl \(1993)|Comedy&#124;Drama|
|17|307|Three Colors: Blue \(Trois couleurs: Bleu) (1993)|Drama|
|18|122|Boomerang \(1992)|Comedy&#124;Romance|
|19|498|Mr\. Jones (1993)|Drama&#124;Romance|

12 movies overlap between the two "popular 100" lists. 

We did not include a damping factor, as the produced popular 100 movies are already films with many ratings. 

### Baseline Evaluation 
MAP stands for Mean Average Precision

|                  | Small Test Set     | Large Test Set     | Small Validation Set | Large Validation Set |
| ---------------- | ------------------ | ------------------ | -------------------- | -------------------- |
| Precision at 15  | 0.0375956284153005 | 0.0260351671884753 | 0.0424043715846994   | 0.0260221010843159   |
| Precision at 25  | 0.0319999999999999 | 0.0191833139659677 | 0.0354098360655737   | 0.0191603958049416   |
| Precision at 100 | 0.0121639344262295 | 0.0069795051329687 | 0.0133770491803278   | 0.0069108105705990   |
| MAP at 15        | 0.0135201349955448 | 0.0075160385323441 | 0.0180145658834183   | 0.0075601142847988   |
| MAP at 25        | 0.0096144119924909 | 0.0048957467170430 | 0.0122820741247914   | 0.0049200717696348   |
| MAP at 100       | 0.0026832507184029 | 0.0013482747269400 | 0.0033829110034699   | 0.0013484062376329   |

We aren't conclusive with our evaluation metrics choice yet. We will do more research and choose the most applicable to ALS. 
## Preliminary Matrix Factorization 

We attempted the Alternating Least Square algorithm and produced a sample result of top 5 recommendations for 10 random users. 

```
+------+----------------------------------------------------------------------------------------------+
|userId| top 5 recommendations[movie, score]
+------+----------------------------------------------------------------------------------------------+
|471   |[[222, 9.660829], [674, 9.521336], [71745, 9.412183], [56145, 9.3069725], [5419, 9.104151]    |
|463   |[[53123, 6.713231], [2517, 6.617906], [8633, 6.604064], [1096, 6.538463], [2867, 6.4077063]   |
|496   |[[1218, 8.731694], [84944, 8.0063095], [7318, 7.759839], [7669, 7.63426], [7121, 7.4429455]   |
|148   |[[6975, 6.394817], [7982, 6.015385], [5504, 5.8545194], [48783, 5.5334735], [1011, 5.461198]  |
|540   |[[3266, 7.267996], [52435, 6.572324], [3089, 6.3977733], [1172, 6.328927], [4144, 6.281991]   |
|392   |[[181, 9.242445], [4941, 8.414173], [3477, 8.331329], [7160, 7.831151], [1078, 7.5596266]     |
|243   |[[37384, 9.74658], [7121, 9.620366], [5528, 9.59763], [72171, 9.455458], [3682, 9.347621]     |
|31    |[[3477, 10.046748], [7247, 8.480147], [6731, 7.374442], [5015, 7.2267447], [492, 7.118576]    |
|516   |[[37384, 7.457745], [111781, 7.1459284], [4437, 6.935057], [6385, 6.6781235], [6461, 6.461185]|
|580   |[[46, 8.125303], [55276, 7.7667303], [955, 7.2712364], [6548, 7.0627236], [1096, 6.9995127]   |
|251   |[[4437, 7.36029], [194, 7.336542], [1693, 7.3364725], [3270, 7.265298], [215, 7.084586]       |
|451   |[[307, 7.7318306], [3925, 7.7122393], [49932, 7.616297], [3224, 7.604404], [119141, 7.603178] |
|85    |[[3918, 7.1662636], [5690, 7.1059356], [2318, 7.1039724], [3035, 6.751943], [2427, 6.746648]  |
|137   |[[37384, 5.3463163], [6385, 5.2862964], [215, 5.2614284], [5992, 5.1995034], [26131, 5.1799316]|
+------+----------------------------------------------------------------------------------------------+
```
## Files in this Repository 
`ALS.py` basic skeleton code for ALS

`partition.py` data split 

`baseline_evaluation.py` evaluation function for our baseline model 

`Data_inspect` exploring the MovieLens datasets hosted on HPC

`popularity.py` baseline model by popularity 
