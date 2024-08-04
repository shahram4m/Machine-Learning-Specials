# # import collections
#
# k=7
# m=867
# # matrix=[['2', '5', '4'], ['3', '7', '8', '9'], ['5', '5', '7', '8', '9', '10']]
# # i=[4, 5, 7, 8, 9, 10]
# # print(int(max(i)))
# #
# # sum=0
# # for index, item in enumerate(matrix):
# #     print(item)
# #     print(int(max(item)))
# #     sum = sum + int(max(item))**2
# #
# # print(int(sum) % int(m))
#
#
# # print(int(max(i))**2)
# # print(list(filter(i, lambda: x : max(int(x)) )))
#
#
# matrix_str = """7 6429964 4173738 9941618 2744666 5392018 5813128 9452095
# 7 6517823 4135421 6418713 9924958 9370532 7940650 2027017
# 7 1506500 3460933 1550284 3679489 4538773 5216621 5645660
# 7 7443563 5181142 8804416 8726696 5358847 7155276 4433125
# 7 2230555 3920370 7851992 1176871 610460 309961 3921536
# 7 8518829 8639441 3373630 5036651 5291213 2308694 7477960
# 7 7178097 249343 9504976 8684596 6226627 1055259 4880436"""
#
# # Split the matrix by rows and convert elements to integers
# matrix_int = [list(map(int, row.split())) for row in matrix_str.split('\n')]
# print(matrix_int)
#
# # Convert matrix elements to integer
# # matrix_int = [[int(i) for i in sublist] for sublist in matrix]
# sum=0
# for index, item in enumerate(matrix_int):
#     print(max(item))
#     sum = sum + (max(item))
#
# print('sum', sum)
# print(sum % m)
#


#print('E', 533370065617465 % 867)

#
# # import os
# # import re
# import collections
#
#
# def getSpamEmails(subjects, spam_words):
#     spam_set = set(word.lower() for word in spam_words)
#     # Convert subject to lowercase and split into a list of words
#     result = []
#     for index, item in enumerate(subjects):
#         subject_words = item.lower().split()
#         word_count = collections.Counter(subject_words)
#         print(word_count)
#         print(spam_set)
#         print(sum)
#         spam_word_count = sum(word_count[word] for word in spam_set)
#         if spam_word_count >= 2:
#             result.append('spam')
#         else:
#             result.append('not_spam')
#
#     return result
#
#     # print(set(spam.lower() for spam in spam_words))
#
#     # # Write your code here
#     # result= []
#     # for index, item in enumerate(subjects):
#     #     sub = set( item.lower().rstrip().split(' '))
#     #     print(sub)
#
#     #     if len(sub.intersection(set(spam.lower() for spam in spam_words) ))>=2:
#     #         result.append('spam')
#     #     else:
#     #         result.append('not_spam')
#     # return result
#
#
# # gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd
# # gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd
# # alcgxovldqfzaor hdigyojknvi ztpcmxlvovafh phvshyfiqqtqbxjj qngqjhwkcexec dkmzakbzrkjwqdy
# # gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd
# # ssjoatryxmbwxbw xnagmaygz fnzpqftobtaotua xmwvzllkujidh kzwzcltgqngguft ahalwvjwqncksiz
# # gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd
# # 11
# # gpuamkxkszhkbpph
# # kinkezplvfjaq
# # opodo
# # krjz
# # imlvumuare
# # excfyc
# # beurg
# # jyos
# # dhvuyfvtvn
# # dyluacvray
# # Gwnpnzijd
#
#
# subjects = ["gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd gwnpnzijd",
#             "alcgxovldqfzaor", "hdigyojknvi", "ztpcmxlvovafh", "phvshyfiqqtqbxjj", "qngqjhwkcexec", "dkmzakbzrkjwqdy",
#             "gwnpnzijd", "gwnpnzijd", "gwnpnzijd", "gwnpnzijd", "gwnpnzijd", "gwnpnzijd", "gwnpnzijd", "gwnpnzijd",
#             "gwnpnzijd",
#             "ssjoatryxmbwxbw", "xnagmaygz", "fnzpqftobtaotua", "xmwvzllkujidh", "kzwzcltgqngguft", "ahalwvjwqncksiz",
#             "gwnpnzijd", "gwnpnzijd", "gwnpnzijd", "gwnpnzijd", "gwnpnzijd", "gwnpnzijd", "gwnpnzijd", "gwnpnzijd",
#             "gwnpnzijd"]
# spam_words = ["free", "monry", "win", "millions", "Gwnpnzijd"]
#
# print(getSpamEmails(subjects, spam_words))
#

# import requests
# def getDiscountedPrice(barcode):
#     # Write your code here
#     url = f"https://jsonmock.hackerrank.com/api/inventory?barcode={barcode}"
#     response = requests.request("GET", url)
#     if len(response.json()['data'])>0:
#         discount = response.json()['data'][0]['discount']
#         price = response.json()['data'][0]['price']
#         discountPrice = round(price - ((discount/100) * price))
#         return discountPrice
#     else:
#         return 0
#
# print(getDiscountedPrice('74005364'))



# data = [9941618, 9924958, 5645660, 8804416, 7851992, 8639441, 9504976]
# # Calculate the sum of the squares of the elements in the data list
# sum_of_squares = sum( pow(x,2,867) for x in data)
# print(sum_of_squares)
#
# # Calculate the result of the sum of squares modulo 867
# result2 = sum_of_squares % 867
# print(result2)

