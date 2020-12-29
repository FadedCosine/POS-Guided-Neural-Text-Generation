# import itertools
# import regex as re
# import jamotools

# try:
#     import mecab
#     Mecab = mecab.MeCab
# except ModuleNotFoundError:
#     from konlpy.tag import Mecab

# def space_normalize(text):
#     return re.sub(' +', ' ', text.strip())

# class Mecab_Tokenizer:
#     def __init__(self,space_symbol='‐', jamo=False):
#         self.space_symbol = space_symbol
#         self.tokenizer = Mecab()
#         self.space_symbol = space_symbol
#         self.jamo = jamo

#     def text_to_morphs(self, text, to_string=False):
#         text = space_normalize(text)
#         try:
#             morphs = self.tokenizer.morphs(text)
#         except ValueError:
#             morphs = ''
#         if morphs:
#             res = [self.space_symbol+jamotools.split_syllables(morphs[0])] if self.jamo else [self.space_symbol+morphs[0]]
#             text = text[len(morphs[0]):]
#             for i in morphs[1:]:
#                 if text[0] == ' ':
#                     if self.jamo:
#                         res.append(self.space_symbol + jamotools.split_syllables(i))
#                     else:
#                         res.append(self.space_symbol+i)
#                     text = text[len(i)+1:]
#                 else:
#                     if self.jamo:
#                         res.append(jamotools.split_syllables(i))
#                     else:
#                         res.append(i)
#                     text = text[len(i):]
#             if to_string:
#                 return ' '.join(res)
#             else:
#                 return res
#         elif to_string:
#             return ''
#         else:
#             return []

#     def morphs_to_text(self,morph):
#         if isinstance(morph,str):
#             temp = morph.replace('##','').replace(' ','').replace(self.space_symbol,' ').strip()
#             if self.jamo:
#                 return jamotools.join_jamos(temp)
#             else:
#                 return temp
#         else:
#             temp = re.sub(self.space_symbol, ' ', ''.join(morph)).strip()
#             if self.jamo:
#                 return jamotools.join_jamos(temp)
#             else:
#                 return temp