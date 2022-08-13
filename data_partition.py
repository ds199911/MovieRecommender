import pandas as pd

def main(path):
    rating = pd.read_csv(path)

    split_index = []
    for user in rating.userId.unique():
        user_rating = rating[rating.userId == user]
        num = int(user_rating.shape[0]*0.3)
        split_index.append(user_rating.sort_values('timestamp', ascending=False).iloc[0:num , :].index)


    index = len(split_index)//2
    val_index = split_index[0:index]
    test_index = split_index[index:]

    val_index_flat = []
    for user_index in val_index:
        for num in user_index:
            val_index_flat.append(num)

    test_index_flat = []
    for user_index in test_index:
        for num in user_index:
            test_index_flat.append(num)

    train_index_flat = []
    exclude = set(val_index_flat + test_index_flat)
    for element in rating.index.to_list():
        if element not in exclude:
            train_index_flat.append(element)

    val = rating[rating.index.isin(val_index_flat)]
    test = rating[rating.index.isin(test_index_flat)]
    train = rating[rating.index.isin(train_index_flat)]

#     val.to_csv("val_small.csv")
#     test.to_csv("test_small.csv")
#     train.to_csv("train_small.csv")
    val.to_csv("val_large.csv")
    test.to_csv("test_large.csv")
    train.to_csv("train_large.csv")
    
    print("val size:", val.shape[0], "test size:", test.shape[0], "train size:", train.shape[0])
   
if __name__ == "__main__":
    ratings_path = "ratings.csv"
    main(ratings_path)
