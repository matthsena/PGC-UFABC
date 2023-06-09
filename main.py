from crawler import get_top_100_articles, get_all_img, download_img
import common_fn
# import canny
import pearson

# top_100_articles = get_top_100_articles()

# for article in top_100_articles:
#     imgs = get_all_img(article['href'])

#     for img in imgs:
#         download_img(img, f"{article['ranking']}-{article['text']}")

sub_dir = common_fn.get_img_folders()

for sub in sub_dir:
    file_list = common_fn.get_img_files(f'./img/{sub}')
    # comb = common_fn.all_combinations(file_list)

    # print(comb)
    for file in file_list:
        c = [(file, item) for item in file_list]
       
        for c_ in c:
            similarity = pearson.similarity(f'./img/{sub}/{c_[0]}', f'./img/{sub}/{c_[1]}')
            print(f'{similarity} -> {sub}/{c_[0]} - {sub}/{c_[1]}')